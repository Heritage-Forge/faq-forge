import pytest
from pydantic import ValidationError
from typing import List, Dict, Any

from src.validator import (
    FAQItem,
    validate_faq_data,
    validate_faq_data_with_report,
    FAQValidationResult
)

class TestFAQItem:
    def test_valid_faq_item(self):
        """Test that valid FAQ items are created properly"""
        item = FAQItem(question="What is this?", answer="This is a test.")
        assert item.question == "What is this?"
        assert item.answer == "This is a test."
        assert item.category is None

    def test_with_category(self):
        """Test that category field works correctly"""
        item = FAQItem(question="What is this?", answer="This is a test.", category="General")
        assert item.category == "General"

    def test_empty_question(self):
        """Test that empty questions raise validation errors"""
        with pytest.raises(ValidationError) as excinfo:
            FAQItem(question="", answer="This is a test.")
        assert "question cannot be empty" in str(excinfo.value).lower()

    def test_empty_answer(self):
        """Test that empty answers raise validation errors"""
        with pytest.raises(ValidationError) as excinfo:
            FAQItem(question="What is this?", answer="")
        assert "answer cannot be empty" in str(excinfo.value).lower()

    def test_whitespace_only(self):
        """Test that whitespace-only fields are considered empty"""
        with pytest.raises(ValidationError) as excinfo:
            FAQItem(question="What is this?", answer="   ")
        assert "answer cannot be empty" in str(excinfo.value).lower()

    def test_whitespace_stripping(self):
        """Test that whitespace is stripped from valid inputs"""
        item = FAQItem(question="  What is this?  ", answer="  This is a test.  ")
        assert item.question == "What is this?"
        assert item.answer == "This is a test."


class TestValidateFAQData:
    def test_valid_list(self):
        """Test validation with a list of valid items"""
        data = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2", "category": "Cat1"}
        ]
        result = validate_faq_data(data)
        assert len(result) == 2
        assert isinstance(result[0], FAQItem)
        assert result[1].category == "Cat1"

    def test_empty_list(self):
        """Test validation with an empty list"""
        result = validate_faq_data([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_invalid_item(self):
        """Test validation with an invalid item"""
        data = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "", "answer": "A2"}
        ]
        with pytest.raises(ValidationError) as excinfo:
            validate_faq_data(data)
        assert "question cannot be empty" in str(excinfo.value).lower()

    def test_non_list_input(self):
        """Test that non-list inputs raise TypeError"""
        with pytest.raises(TypeError):
            validate_faq_data("not a list")
            
        with pytest.raises(TypeError):
            validate_faq_data({"key": "value"})

    def test_extra_fields(self):
        """Test that extra fields are ignored"""
        data = [{"question": "Q1?", "answer": "A1", "extra_field": "value"}]
        result = validate_faq_data(data)
        assert len(result) == 1
        assert not hasattr(result[0], "extra_field")


class TestValidateFAQDataWithReport:
    def test_mixed_valid_invalid(self):
        """Test reporting with a mix of valid and invalid items"""
        data = [
            {"question": "Q1?", "answer": "A1"},  # valid
            {"question": "", "answer": "A2"},     # invalid
            {"question": "Q3?", "answer": "A3"}   # valid
        ]
        result = validate_faq_data_with_report(data)
        assert isinstance(result, FAQValidationResult)
        assert len(result.valid_items) == 2
        assert len(result.invalid_items) == 1
        assert len(result.validation_errors) == 1
        assert result.invalid_items[0] == {"question": "", "answer": "A2"}

    def test_all_valid(self):
        """Test reporting with all valid items"""
        data = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"}
        ]
        result = validate_faq_data_with_report(data)
        assert len(result.valid_items) == 2
        assert len(result.invalid_items) == 0
        assert len(result.validation_errors) == 0

    def test_all_invalid(self):
        """Test reporting with all invalid items"""
        data = [
            {"question": "", "answer": "A1"},
            {"question": "Q2?", "answer": ""}
        ]
        result = validate_faq_data_with_report(data)
        assert len(result.valid_items) == 0
        assert len(result.invalid_items) == 2
        assert len(result.validation_errors) == 2

    def test_missing_fields(self):
        """Test validation with missing required fields"""
        data = [
            {"question": "Q1?"},  # missing answer
            {"answer": "A2"}      # missing question
        ]
        result = validate_faq_data_with_report(data)
        assert len(result.valid_items) == 0
        assert len(result.invalid_items) == 2
        assert "missing" in result.validation_errors[0].lower()

    def test_edge_case_unicode(self):
        """Test with unicode characters"""
        data = [{"question": "¿Cómo estás?", "answer": "Estoy bien, ¡gracias!"}]
        result = validate_faq_data_with_report(data)
        assert len(result.valid_items) == 1
        assert result.valid_items[0].question == "¿Cómo estás?"

    def test_edge_case_long_content(self):
        """Test with extremely long content"""
        long_text = "a" * 10000
        data = [{"question": "Is this a long answer?", "answer": long_text}]
        result = validate_faq_data_with_report(data)
        assert len(result.valid_items) == 1
        assert len(result.valid_items[0].answer) == 10000