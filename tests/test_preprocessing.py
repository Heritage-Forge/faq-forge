import os
import json
import pandas as pd
import pytest
from unittest.mock import patch, mock_open
from src.preprocessing import DataPreprocessor

RAW_DATA = [
    {
        "question": "  Do you sell 3D-printed parts for 1967 Ford Mustangs?  ",
        "answer": "Yes, we offer precision-engineered 3D printed parts for 1965–1970 Mustangs."
    },
    {
        "question": "Do you sell 3D-printed parts for 1967 Ford Mustangs?",
        "answer": "Yes, we offer precision-engineered 3D printed parts for 1965–1970 Mustangs."
    },
    {
        "question": "<p>What's the expected delivery time for custom suspension components?</p>",
        "answer": "Lead time is typically 2–4 weeks depending on production load.",
        "category": "Delivery ,"
    }
]

INVALID_DATA = [
    {
        "question": "  Do you sell 3D-printed parts for 1967 Ford Mustangs?  ",
        "answer": ""  # Empty answer
    },
    {
        "question": "",  # Empty question
        "answer": "Yes, we offer precision-engineered 3D printed parts."
    },
    {
        "answer": "This is missing a question"  # Missing question field
    }
]

@pytest.fixture
def sample_input_file(tmp_path):
    file_path = tmp_path / "raw_faq.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(RAW_DATA, f)
    return file_path

@pytest.fixture
def invalid_input_file(tmp_path):
    file_path = tmp_path / "invalid_faq.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(INVALID_DATA, f)
    return file_path

@pytest.fixture
def output_file(tmp_path):
    return tmp_path / "cleaned_faq.json"

@pytest.fixture
def report_file(tmp_path):
    return tmp_path / "validation_report.json"

class TestDataPreprocessor:
    def test_clean_text(self):
        """Test the clean_text method with various inputs."""
        preprocessor = DataPreprocessor("data/faq_parts.json", "dummy_out.json")
        
        # Test HTML removal
        assert preprocessor.clean_text("<p>Test</p>") == "test"
        
        # Test whitespace handling
        assert preprocessor.clean_text("  Multiple    spaces  ") == "multiple spaces"
        
        # Test newlines
        assert preprocessor.clean_text("Line 1\nLine 2") == "line 1 line 2"
        
        # Test non-string input
        assert preprocessor.clean_text(None) == ""
        
        # Test mixed case
        assert preprocessor.clean_text("MiXeD cAsE") == "mixed case"

        # Test category handling
        assert preprocessor.clean_text("Deviation  ,  acAsE") == "deviation , acase"

    def test_load_data_file_not_found(self):
        """Test load_data with non-existent file."""
        preprocessor = DataPreprocessor("nonexistent.json", "dummy_out.json")
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data()

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"invalid": "json"')
    def test_load_data_invalid_json(self, mock_file, mock_exists):
        """Test load_data with invalid JSON."""
        preprocessor = DataPreprocessor("data/faq_parts.json", "dummy_out.json")
        with pytest.raises(json.JSONDecodeError):
            preprocessor.load_data()

    def test_full_preprocessing_pipeline(self, sample_input_file, output_file):
        """Test the complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(str(sample_input_file), str(output_file))
        df = preprocessor.run()
        
        assert df.iloc[0]["question"] == "do you sell 3d-printed parts for 1967 ford mustangs?"
        assert "<p>" not in df.iloc[1]["question"]
        
        assert len(df) == 2
        assert os.path.exists(output_file)
        
        with open(output_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert len(saved_data) == 2

    def test_validation_report(self, invalid_input_file, output_file, report_file):
        """Test validation reporting with invalid data."""
        preprocessor = DataPreprocessor(
            str(invalid_input_file), 
            str(output_file),
            report_invalid=True,
            report_filepath=str(report_file)
        )
        
        df = preprocessor.run()
        
        assert df.empty
        
        assert report_file.exists()
        
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
            assert report["summary"]["total_items"] == 3
            assert report["summary"]["valid_items"] == 0
            assert report["summary"]["invalid_items"] == 3
            assert len(report["errors"]) == 3

    def test_empty_dataframe_handling(self, tmp_path):
        """Test handling of empty DataFrame."""
        input_file = tmp_path / "empty.json"
        output_file = tmp_path / "empty_out.json"
        
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("[]")
        
        preprocessor = DataPreprocessor(str(input_file), str(output_file))
        df = preprocessor.run()
        
        assert df.empty
        assert not output_file.exists()

    @patch("pandas.DataFrame.to_json")
    def test_save_clean_data_empty_df(self, mock_to_json):
        """Test that save_clean_data doesn't call to_json with empty DataFrame."""
        preprocessor = DataPreprocessor("data/faq_parts.json", "dummy_out.json")
        empty_df = pd.DataFrame()
        
        preprocessor.save_clean_data(empty_df)
        
        mock_to_json.assert_not_called()
