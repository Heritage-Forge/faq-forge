from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional, Dict, Any

class FAQItem(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None
    
    @field_validator("question", "answer")
    @classmethod
    def not_empty(cls, value: str, info) -> str:
        if not value or not value.strip():
            raise ValueError(f"{info.field_name} cannot be empty.")
        return value.strip()  # Strip whitespace from inputs

class FAQValidationResult(BaseModel):
    """Model to hold validation results"""
    valid_items: List[FAQItem]
    invalid_items: List[Dict[str, Any]]
    validation_errors: List[str]

def validate_faq_data(faq_data: List[dict]) -> List[FAQItem]:
    """
    Validate each FAQ record using the FAQItem model.
    Raises validation errors if any record is invalid.
    
    Args:
        faq_data: List of dictionaries containing FAQ items
        
    Returns:
        List of validated FAQItem objects
        
    Raises:
        ValueError: If any item fails validation
    """
    if not isinstance(faq_data, list):
        raise TypeError("Input must be a list of dictionaries")
    
    return [FAQItem(**item) for item in faq_data]

def validate_faq_data_with_report(faq_data: List[dict]) -> FAQValidationResult:
    """
    Validate FAQ items and return a report including both valid and invalid items.
    Unlike validate_faq_data, this function doesn't raise exceptions for invalid items.
    
    Args:
        faq_data: List of dictionaries containing FAQ items
        
    Returns:
        FAQValidationResult containing valid items, invalid items, and error messages
    """
    if not isinstance(faq_data, list):
        raise TypeError("Input must be a list of dictionaries")
    
    valid_items = []
    invalid_items = []
    validation_errors = []
    
    for i, item in enumerate(faq_data):
        try:
            valid_items.append(FAQItem(**item))
        except ValidationError as e:
            invalid_items.append(item)
            validation_errors.append(f"Item at index {i} failed validation: {str(e)}")
    
    return FAQValidationResult(
        valid_items=valid_items,
        invalid_items=invalid_items,
        validation_errors=validation_errors
    )