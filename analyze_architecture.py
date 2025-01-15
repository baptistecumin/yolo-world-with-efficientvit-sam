import io
import base64
from typing import List, Tuple
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyAcJ9vmQhxzI_aCFwIxmedybQF6NRUlecY")

def analyze_architectural_elements(
    labeled_image: Image.Image,
    model: str = "gemini-1.5-flash",
    prompt: str = (
        "You are an interior design expert analyzing a room image with numbered regions. "
        "Your task is to identify which numbered elements are PERMANENT architectural features "
        "that would require construction work to remove (e.g., built-in cabinets, doors, fireplaces, built-in shelving, walls, windows, etc.). \n\n"
        "Important guidelines:\n"
        "- If an element can be removed without construction work, it is considered movable furniture\n"
        "- ALL artwork and wall decorations are considered movable, even if mounted\n"
        "- ALL light fixtures are considered movable, including ceiling lights and sconces\n"
        "- Built-in cabinets, fireplaces, and permanent shelving ARE architectural features\n"
        "- If you are unsure about an element, consider it movable\n\n"
        "Use the select_architectural_elements tool to specify ONLY the numbers that correspond "
        "to permanent, non-movable architectural features. Only include numbers you are 100% certain "
        "represent elements that require construction to remove."
    )
) -> Tuple[List[int], str]:
    """
    Uses Gemini to determine which numbered masks represent permanent architectural
    features that would require construction to remove, excluding all movable furniture
    and decorative elements.
    
    Returns: (numbers_to_remove, explanation)
    """
    architectural_tool = {
        'function_declarations': [{
            'name': 'select_architectural_elements',
            'description': (
                "Select which numbered regions represent permanent architectural features "
                "that would require construction work to remove. Only include numbers you "
                "are 100% certain about."
            ),
            'parameters': {
                'type': 'OBJECT',
                'properties': {
                    'explanation': {
                        'type': 'STRING',
                        'description': (
                            "Detailed explanation of which items are permanent architectural "
                            "features and why they require construction to remove."
                        ),
                    },
                    'numbers_to_remove': {
                        'type': 'array',
                        'description': (
                            "List of integer IDs corresponding to permanent architectural "
                            "elements that require construction to remove."
                        ),
                        'items': {
                            'type': 'number',
                        },
                    },
                },
                'required': ['numbers_to_remove', 'explanation']
            }
        }]
    }

    # Convert the labeled PIL image to base64
    labeled_image = labeled_image.resize((512, 512))  # clamp size for consistency
    buf = io.BytesIO()
    labeled_image.save(buf, format="JPEG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    parts = [
        {'mime_type': 'image/jpeg', 'data': b64_image},
        prompt
    ]

    try:
        model_instance = genai.GenerativeModel(model_name=model, tools=architectural_tool)
        chat = model_instance.start_chat()
        response = chat.send_message(parts)
        print("[Architectural] Response from Gemini:\n", response)
        
        # Parse function call
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                continue
            for part in candidate.content.parts:
                if not hasattr(part, 'function_call'):
                    continue
                fn_call = part.function_call
                if not hasattr(fn_call, 'args'):
                    continue
                args_dict = fn_call.args
                if args_dict is None:
                    continue
                numbers = list(args_dict.get('numbers_to_remove', []))
                explanation = str(args_dict.get('explanation', ''))
                return (numbers, explanation)

        print("[Architectural] No valid function call found in response.")
        return ([], "No valid function call found in Gemini response.")

    except Exception as e:
        print("[Architectural] Error calling Gemini:", e)
        raise

    return ([], "Unexpected error occurred.")