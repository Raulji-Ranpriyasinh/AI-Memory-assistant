"""
Food image recognition service (Phase 6).
Wrapper pattern — backend implementation is swappable without changing the API contract.
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from typing import List, Optional

import google.generativeai as genai
from pydantic import BaseModel


# ── Result model ─────────────────────────────────────────────────────────────

class FoodRecognitionResult(BaseModel):
    items: List[str]
    estimated_calories: int
    glycemic_load: str  # 'low' | 'medium' | 'high'
    portion_sizes: List[str]
    confidence: float  # 0-1
    # Extended fields from Gemini response
    portion_description: Optional[str] = None
    portion_weight_grams: Optional[int] = None
    glycemic_index: Optional[dict] = None
    nutrients: Optional[dict] = None
    micronutrients: Optional[dict] = None
    health_score: Optional[int] = None
    health_notes: Optional[str] = None
    suitable_for: Optional[List[str]] = None
    caution_for: Optional[List[str]] = None


# ── Abstract service ────────────────────────────────────────────────────────

class FoodRecognitionService(ABC):
    """Abstract base class for food recognition backends."""

    @abstractmethod
    def recognize(self, image_base64: str) -> FoodRecognitionResult:
        """
        Recognize food from a base64-encoded image.
        Returns a FoodRecognitionResult.
        """
        ...


# ── Configuration error ─────────────────────────────────────────────────────

class ConfigurationError(Exception):
    """Raised when no food recognition backend can be configured."""
    pass


# ── Gemini Vision recognizer ────────────────────────────────────────────────

class GeminiVisionRecognizer(FoodRecognitionService):
    """
    Uses Gemini Vision API to analyze food images.
    Sends image with structured prompt asking for:
    food items list, portion sizes, estimated calories, glycemic load.
    Parses JSON response.
    """

    FOOD_ANALYSIS_PROMPT = """
You are a nutrition expert and dietitian. Analyze the food in this image and provide a detailed nutritional breakdown.

Return your response as a valid JSON object with the following structure:
{
  "food_items": ["list of identified food items"],
  "estimated_portion": {
    "description": "e.g., 1 medium plate, 1 cup, 200g",
    "weight_grams": 0
  },
  "glycemic_index": {
    "value": 0,
    "category": "Low (< 55) | Medium (55-69) | High (≥ 70) | N/A",
    "notes": "brief explanation"
  },
  "glycemic_load": {
    "value": 0,
    "category": "Low (< 10) | Medium (10-20) | High (> 20) | N/A"
  },
  "nutrients_per_serving": {
    "calories": 0,
    "carbohydrates_g": 0,
    "protein_g": 0,
    "fat_g": 0,
    "fiber_g": 0,
    "sugar_g": 0,
    "sodium_mg": 0
  },
  "micronutrients": {
    "vitamins": ["list of notable vitamins present"],
    "minerals": ["list of notable minerals present"]
  },
  "health_score": 0,
  "health_notes": "brief overall health assessment",
  "suitable_for": ["e.g., diabetics, weight loss, athletes"],
  "caution_for": ["e.g., high blood pressure, diabetes"],
  "confidence": 0.85
}

Be as accurate as possible based on visual cues. If uncertain, provide reasonable estimates.
Return ONLY the JSON object, no extra text.
"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def recognize(self, image_base64: str) -> FoodRecognitionResult:
        """
        Decode base64 image, send to Gemini Vision, parse JSON response.
        """
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)

        response = self.model.generate_content([
            {"mime_type": "image/jpeg", "data": image_bytes},
            self.FOOD_ANALYSIS_PROMPT,
        ])

        raw_text = response.text.strip()

        # Strip markdown code fences if present
        raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
        raw_text = re.sub(r"\n?```$", "", raw_text)

        data = json.loads(raw_text)

        # Map glycemic load category
        gl_category_map = {
            "Low (< 10)": "low",
            "Medium (10-20)": "medium",
            "High (> 20)": "high",
            "N/A": "low",
        }
        gl_raw = data.get("glycemic_load", {}).get("category", "low")
        glycemic_load = gl_category_map.get(gl_raw, "low")

        return FoodRecognitionResult(
            items=data.get("food_items", []),
            estimated_calories=int(data.get("nutrients_per_serving", {}).get("calories", 0)),
            glycemic_load=glycemic_load,
            portion_sizes=[data.get("estimated_portion", {}).get("description", "")],
            confidence=float(data.get("confidence", 0.5)),
            # Extended fields
            portion_description=data.get("estimated_portion", {}).get("description"),
            portion_weight_grams=data.get("estimated_portion", {}).get("weight_grams"),
            glycemic_index=data.get("glycemic_index"),
            nutrients=data.get("nutrients_per_serving"),
            micronutrients=data.get("micronutrients"),
            health_score=data.get("health_score"),
            health_notes=data.get("health_notes"),
            suitable_for=data.get("suitable_for", []),
            caution_for=data.get("caution_for", []),
        )


# ── Cloud Vision + Nutritionix recognizer ───────────────────────────────────

class CloudVisionNutritionixRecognizer(FoodRecognitionService):
    """
    Step 1: Google Cloud Vision API → extract food labels.
    Step 2: NutritionAPIClient.lookup_batch(labels) → get nutrition data.
    Combine into FoodRecognitionResult.
    """

    def __init__(self, nutritionix_app_id: str, nutritionix_api_key: str):
        self.nutritionix_app_id = nutritionix_app_id
        self.nutritionix_api_key = nutritionix_api_key

    def recognize(self, image_base64: str) -> FoodRecognitionResult:
        """
        Use Google Cloud Vision to identify food items,
        then query Nutritionix for nutritional data.
        """
        # Step 1: Google Cloud Vision API for label detection
        from google.cloud import vision

        image_bytes = base64.b64decode(image_base64)
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.label_detection(image=image)

        labels = [label.description.lower() for label in response.label_annotations]
        food_labels = [l for l in labels if l in COMMON_FOOD_KEYWORDS]

        if not food_labels:
            # If no food labels found, try a broader approach
            food_labels = labels[:5]

        if not food_labels:
            return FoodRecognitionResult(
                items=[],
                estimated_calories=0,
                glycemic_load="low",
                portion_sizes=[],
                confidence=0.0,
            )

        # Step 2: Nutritionix API lookup
        from app.integrations.nutrition_api import NutritionAPIClient
        nutrition_client = NutritionAPIClient(
            self.nutritionix_app_id, self.nutritionix_api_key
        )
        nutrition_data = nutrition_client.lookup_batch(food_labels)

        # Combine results
        items = food_labels
        total_calories = sum(item.get("nf_calories", 0) or 0 for item in nutrition_data)
        portion_sizes = [
            f"{item.get('brand_name', item.get('food_name', ''))} - {item.get('serving_qty', '')} {item.get('serving_unit', '')}"
            for item in nutrition_data
        ]

        # Estimate glycemic load from carb content
        total_carbs = sum(item.get("nf_total_carbohydrate", 0) or 0 for item in nutrition_data)
        if total_carbs < 10:
            glycemic_load = "low"
        elif total_carbs < 20:
            glycemic_load = "medium"
        else:
            glycemic_load = "high"

        return FoodRecognitionResult(
            items=items,
            estimated_calories=int(total_calories),
            glycemic_load=glycemic_load,
            portion_sizes=portion_sizes,
            confidence=0.7,
        )


# ── Common food keywords for label filtering ────────────────────────────────

COMMON_FOOD_KEYWORDS = {
    'apple', 'banana', 'bread', 'butter', 'cheese', 'chicken', 'chocolate',
    'cookie', 'egg', 'fish', 'fruit', 'juice', 'milk', 'orange', 'pasta',
    'pizza', 'potato', 'rice', 'salad', 'sandwich', 'soup', 'steak', 'vegetable',
    'yogurt', 'coffee', 'tea', 'sugar', 'flour', 'oil', 'rice', 'noodle',
    'burger', 'fries', 'cake', 'pie', 'ice cream', 'cereal', 'oatmeal',
    'beans', 'lentils', 'tofu', 'nuts', 'almonds', 'peanut', 'avocado',
    'tomato', 'carrot', 'onion', 'garlic', 'pepper', 'spinach', 'broccoli',
}


# ── Factory function ────────────────────────────────────────────────────────

def get_food_recognizer() -> FoodRecognitionService:
    """
    Factory function.
    Returns GeminiVisionRecognizer if GEMINI_API_KEY is set,
    else CloudVisionNutritionixRecognizer if NUTRITIONIX_APP_ID is set,
    else raises ConfigurationError.
    """
    import os

    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
    nutritionix_app_id = os.getenv("NUTRITIONIX_APP_ID")
    nutritionix_api_key = os.getenv("NUTRITIONIX_API_KEY")

    if gemini_api_key:
        return GeminiVisionRecognizer(gemini_api_key)
    elif nutritionix_app_id and nutritionix_api_key:
        return CloudVisionNutritionixRecognizer(
            nutritionix_app_id, nutritionix_api_key
        )
    else:
        raise ConfigurationError(
            "No food recognition backend configured. "
            "Set GEMINI_API_KEY or (NUTRITIONIX_APP_ID + NUTRITIONIX_API_KEY)."
        )


# ── Convenience function ────────────────────────────────────────────────────

def recognize_food_image(image_base64: str) -> dict:
    """
    High-level convenience function.
    Returns a dict suitable for API response.
    """
    recognizer = get_food_recognizer()
    result = recognizer.recognize(image_base64)
    return result.model_dump()
