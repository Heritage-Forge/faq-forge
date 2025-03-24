import json
import re
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.validator import FAQItem, validate_faq_data, validate_faq_data_with_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("preprocessing")

class DataPreprocessor:
    """
    Class for preprocessing and cleaning FAQ data.
    
    Handles loading, validating, cleaning, and saving FAQ data.
    """
    
    def __init__(self, input_filepath: str, output_filepath: str, 
                 report_invalid: bool = True, 
                 report_filepath: Optional[str] = None):
        """
        Initialize the preprocessor with file paths.
        
        Args:
            input_filepath: Path to the input JSON file
            output_filepath: Path to save the processed data
            report_invalid: Whether to report invalid items
            report_filepath: Path to save validation report (if None, uses stdout)
        """
        self.input_filepath = Path(input_filepath)
        self.output_filepath = Path(output_filepath)
        self.report_invalid = report_invalid
        self.report_filepath = Path(report_filepath) if report_filepath else None
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean the input text by:
         - Stripping leading/trailing whitespace
         - Collapsing multiple spaces into one
         - Removing HTML tags
         - Converting to lowercase
         
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Strip leading/trailing whitespace
        cleaned = text.strip()

        # Remove HTML tags (if any)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # Collapse multiple spaces/newlines into a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Normalize text to lowercase
        cleaned = cleaned.lower()
        
        return cleaned
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data from the input file.
        
        Returns:
            Raw data as a list of dictionaries
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            json.JSONDecodeError: If input file is not valid JSON
        """
        logger.info(f"Loading data from {self.input_filepath}")
        
        if not self.input_filepath.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_filepath}")
            
        with open(self.input_filepath, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
                logger.info(f"Loaded {len(raw_data)} FAQ items")
                return raw_data
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in input file: {e}")
                raise
    
    def validate_data(self, raw_data: List[Dict[str, Any]]) -> List[FAQItem]:
        """
        Validate the data using Pydantic models.
        
        Args:
            raw_data: Raw FAQ data as list of dictionaries
            
        Returns:
            List of validated FAQItem objects
            
        Raises:
            ValueError: If validation fails and report_invalid is False
        """
        logger.info("Validating FAQ data")
        
        if self.report_invalid:
            validation_result = validate_faq_data_with_report(raw_data)
            valid_items = validation_result.valid_items
            
            # Report invalid items if any
            if validation_result.invalid_items:
                report_msg = (
                    f"Found {len(validation_result.invalid_items)} invalid items "
                    f"out of {len(raw_data)} total items."
                )
                logger.warning(report_msg)
                
                # Save detailed report if requested
                if self.report_filepath:
                    self._save_validation_report(validation_result)
                else:
                    # Log the issues
                    for error in validation_result.validation_errors:
                        logger.warning(f"Validation error: {error}")
            
            return valid_items
        else:
            # This will raise an error if any item is invalid
            return validate_faq_data(raw_data)
    
    def _save_validation_report(self, validation_result) -> None:
        """
        Save validation report to a file.
        
        Args:
            validation_result: Validation result object
        """
        if not self.report_filepath:
            return
            
        report = {
            "summary": {
                "total_items": len(validation_result.valid_items) + len(validation_result.invalid_items),
                "valid_items": len(validation_result.valid_items),
                "invalid_items": len(validation_result.invalid_items)
            },
            "errors": validation_result.validation_errors,
            "invalid_data": validation_result.invalid_items
        }
        
        self.report_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Validation report saved to {self.report_filepath}")
    
    def load_and_validate_data(self) -> List[FAQItem]:
        """
        Load the JSON data from file and validate it.
        
        Returns:
            List of validated FAQItem objects
        """
        raw_data = self.load_data()
        faq_items = self.validate_data(raw_data)
        logger.info(f"Successfully validated {len(faq_items)} FAQ items")
        return faq_items
    
    def process_data(self) -> pd.DataFrame:
        """
        Process the FAQ data:
         - Validate and load data
         - Convert to a DataFrame
         - Apply text cleaning to question, answer, and category
         - Remove duplicate entries
         
        Returns:
            A cleaned pandas DataFrame
        """
        faq_items = self.load_and_validate_data()
        
        df = pd.DataFrame([item.model_dump() for item in faq_items])
        
        if df.empty:
            logger.warning("No valid FAQ items found")
            return df
        
        initial_count = len(df)
        logger.info(f"Cleaning {initial_count} FAQ items")
        
        df["question"] = df["question"].apply(self.clean_text)
        df["answer"] = df["answer"].apply(self.clean_text)
        df["category"] = df["category"].apply(self.clean_text)
        
        df = df.drop_duplicates(subset=["question", "answer", "category"]).reset_index(drop=True)
        
        duplicate_count = initial_count - len(df)
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate entries")
        
        return df
    
    def save_clean_data(self, df: pd.DataFrame) -> None:
        """
        Save the cleaned DataFrame back to a JSON file.
        
        Args:
            df: Cleaned DataFrame to save
        """
        if df.empty:
            logger.warning("No data to save")
            return
            
        self.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(df)} cleaned FAQ items to {self.output_filepath}")
        df.to_json(
            self.output_filepath, 
            orient="records", 
            indent=2, 
            force_ascii=False
        )
        logger.info("Data successfully saved")
    
    def run(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Processed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        try:
            df = self.process_data()
            self.save_clean_data(df)
            logger.info("Preprocessing completed successfully")
            return df
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise