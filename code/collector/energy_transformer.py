import os
import pandas as pd
from datetime import datetime
from config import Config_collector


class RENTransformer:
    """Transform RAW REN electricity data into a clean dataset"""
    def __init__(self):
        self.config = Config_collector()
        self.raw_dir = self.config.DATA_DIR_RAW

        # Target and feature columns expected by the model
        self.target_cols = [
            "date",                    
            "corrigido_temperatura",   # TARGET

            # Energy production 
            "solar",
            "eolica",
            "hidrica",
            "gas_natural",
            "biomassa",
            "outra_termica",

            # International energy exchanges
            "importacao",
            "exportacao",

            # Storage / pumped hydro
            "producao_bombagem",
            "consumo_bombagem",

            # Specific consumption
            "consumo_armazenamento",
        ]

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, sort by date, and enforce date consistency."""
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        df = df.drop_duplicates(subset="date", keep="last")

        return df.reset_index(drop=True)

    def _process_raw_data(self) -> pd.DataFrame:
        """Load and transform RAW REN CSV into a wide-format dataframe."""
        path = os.path.join(self.raw_dir, "ren_consumption_raw.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"RAW file not found: {path}")
        df = pd.read_csv(path, low_memory=False)

        if df.empty:
            raise ValueError("RAW file is empty")

        required_cols = ["date_query", "type", "daily_Accumulation"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in RAW file: {missing_cols}")

        # Normalize date column
        df["date"] = pd.to_datetime(df["date_query"], errors="coerce")
        df = df.dropna(subset=["date", "type", "daily_Accumulation"])
        pivot_df = (
            df.pivot_table(
                index="date",
                columns="type",
                values="daily_Accumulation",
                aggfunc="first",
            )
            .reset_index()
        )

        # Normalize column names
        pivot_df.columns = [
            col if col == "date" else str(col).lower().replace(" ", "_")
            for col in pivot_df.columns
        ]

        return self._clean_dataframe(pivot_df)

    def _select_and_validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select required columns and validate model requirements."""
        available_cols = [c for c in self.target_cols if c in df.columns]
        missing_cols = [c for c in self.target_cols if c not in df.columns]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")

        if "corrigido_temperatura" not in df.columns:
            raise ValueError("TARGET column 'corrigido_temperatura' not found")

        if len(available_cols) < 5:
            print(f"Very few usable columns: {available_cols}")

        selected_df = df[available_cols].copy()

    
        return selected_df

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run basic data quality checks and fixes."""
        if df.empty:
            return df

       

        # Missing values
        missing = df.isnull().sum()
        if missing.any():
            print(" Missing values:")
            for col, cnt in missing[missing > 0].items():
                pct = (cnt / len(df)) * 100
                print(f"   - {col}: {cnt} ({pct:.1f}%)")

            # Median imputation for numeric columns
            for col in df.select_dtypes("number"):
                df[col] = df[col].fillna(df[col].median())

        # Date range
        print(
            f"Data range: {df['date'].min().date()} - {df['date'].max().date()} "
            f"({len(df)} days)"
        )

        return df

    def _save_datasets(self, df: pd.DataFrame):
        """Save clean, training, and latest datasets."""
        df.to_csv(self.config.REN_PRODUCTION_PATH, index=False)
        df.to_csv(self.config.REN_DATASET_PATH, index=False)
        latest_df = df.tail(1).copy()
        latest_df.to_csv(self.config.REN_LATEST_PATH, index=False)
        return df, df, latest_df

    def process(self):
        """Full REN transformation pipeline."""
        try:
            raw_df = self._process_raw_data()
            selected_df = self._select_and_validate_columns(raw_df)
            train_df, production_df, latest_df = self._save_datasets(selected_df)
            return train_df, production_df, latest_df

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
