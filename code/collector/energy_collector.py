import requests
import pandas as pd
import time

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path

from config import Config_collector

# Progress control (thread-safe)
progress_lock = Lock()
progress_counter = {
    "completed": 0,
    "total": 0,
    "start_time": 0
}

class RENRawCollector:
    """Collects RAW REN electricity data incrementally."""
    def __init__(self):
        self.Config = Config_collector()
        self.base_url = "https://servicebus.ren.pt/datahubapi/electricity"
        self.culture = "pt-PT"

        self.raw_dir = Path(self.Config.DATA_DIR_RAW)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.endpoints = {
            "electricity": {
                "url": "ElectricityConsumptionSupplyDaily",
                "file": self.raw_dir / "ren_consumption_raw.csv"
            }
        }
    def _last_date(self, filepath: Path):
            """Safely detect last date in a RAW file."""
            if not filepath.exists():
                return None
            try:
                df = pd.read_csv(filepath, usecols=["date_query"], low_memory=False)
                if df.empty:
                    return None
                last_date_str = df.iloc[-1]["date_query"]
                return datetime.strptime(last_date_str, "%Y-%m-%d")
            except Exception as e:
                return None

    def _get_daily(self, endpoint: dict, date: str) -> pd.DataFrame:
        """Download one day of data from REN API."""
        url = f"{self.base_url}/{endpoint['url']}"
        params = {"culture": self.culture, "date": date}

        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code != 200:
                print(f"Status {r.status_code} para {date}")
                return pd.DataFrame()

            data = r.json()
            if not data or not isinstance(data, list):
                return pd.DataFrame()
            rows = []
            for item in data:
                if isinstance(item, dict) and "type" in item:
                    type_name = item["type"].lower()
                    daily_value = item.get("daily_Accumulation")
                    
                    if daily_value is not None:
                        rows.append({
                            "date_query": date,
                            "type": type_name,
                            "daily_Accumulation": float(daily_value)
                        })
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            return pd.DataFrame()

    def print_progress(self, label: str):
        completed = progress_counter["completed"]
        total = progress_counter["total"]

        if total == 0:
            return

        elapsed = time.time() - progress_counter["start_time"]
        pct = (completed / total) * 100
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (total - completed) / rate if rate > 0 else 0

        bar_len = 40
        filled = int(bar_len * completed / total)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{label}] [{bar}] {pct:5.1f}% | "
            f"{completed}/{total} days | "
            f"{elapsed/60:.1f}m/{remaining/60:.1f}m",
            end="",
            flush=True
        )
    
    def update(self, end_date: str | None = None, max_workers: int = 6):
        """Incrementally update RAW REN datasets.
        """

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        for key, endpoint in self.endpoints.items():
            path = endpoint["file"]
            print(f"\nUpdating {key.upper()}...")

            # Get last available date in the file
            last_date = self._last_date(path)

            if last_date is None:
                start_date = self.Config.START_DATE
            else:
                # Start from the day after the last recorded date
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

            # Convert to datetime for comparison
            start_dt = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if isinstance(start_date, str)
                else start_date
            )
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            if start_dt > end_dt:
                print(f"Data already up to date until {end_date}")
                continue

            # Generate list of dates to download
            dates = pd.date_range(start=start_dt, end=end_dt, freq="D")

            if len(dates) == 0:
                print("No new dates to download")
                continue

            print(f" Downloading {len(dates)} days ({start_date} : {end_date})")

            # Initialize progress tracking
            with progress_lock:
                progress_counter["completed"] = 0
                progress_counter["total"] = len(dates)
                progress_counter["start_time"] = time.time()

            frames = []

            # Parallel downloads using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._get_daily,
                        endpoint,
                        date.strftime("%Y-%m-%d")
                    ): date
                    for date in dates
                }

                for future in as_completed(futures):
                    date_obj = futures[future]
                    df = future.result()

                    if not df.empty:
                        frames.append(df)

                    # Update progress
                    with progress_lock:
                        progress_counter["completed"] += 1
                        self.print_progress(key.upper())

            print()  # New line after progress bar

            # Combine downloaded data
            if not frames:
                print(f" No data downloaded for {key.upper()}")
                continue

            df_new = pd.concat(frames, ignore_index=True)

            # Merge with existing file if it exists
            if path.exists():
                try:
                    df_old = pd.read_csv(path, low_memory=False)
                    df_final = pd.concat([df_old, df_new], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing file: {e}")
                    df_final = df_new
            else:
                df_final = df_new

            # Remove duplicates (keep the most recent record)
            df_final = df_final.drop_duplicates(
                subset=["date_query", "type"],
                keep="last"
            )

            # Sort by date and type
            df_final = (
                df_final
                .sort_values(["date_query", "type"])
                .reset_index(drop=True)
            )

            # Save updated dataset
            df_final.to_csv(path, index=False)
            print(f"{key.upper()} saved: {len(df_final)} records : {path}")

            # Small pause between endpoints
            time.sleep(1)

        print("\nUpdate completed successfully!")
