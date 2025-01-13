import backoff
from json.decoder import JSONDecodeError
from typing import List, Tuple
from PIL import Image
import base64
import io
import google.generativeai as genai
genai.configure(api_key="AIzaSyAcJ9vmQhxzI_aCFwIxmedybQF6NRUlecY")

@backoff.on_exception(
    backoff.expo,
    (JSONDecodeError,),
    max_tries=2,
    max_time=60,
)
def analyze_architectural_elements(
    labeled_image: Image.Image,
    model: str = "gemini-1.5-flash",
    prompt: str = (
        "Analyze this room image with letter-labeled regions. Identify which letters"
        " correspond to permanent architectural features (walls, built-ins, windows)."
        " Use the select_architectural_elements tool to specify letters to remove from"
        " the furniture segmentation since they're not removable furniture."
    )
) -> Tuple[List[str], str]:
    """
    Uses Gemini to determine which lettered masks should be removed because they're
    permanent architectural features, not removable furniture.
    
    Returns: (letters_to_remove, explanation).
    """
    # The tool spec for function calling
    architectural_tool = {
        'function_declarations': [{
            'name': 'select_architectural_elements',
            'description': (
                "Select which numbered regions represent permanent architectural features "
                "that would require construction work to remove. These are elements that "
                "are built into the structure of the room, not merely heavy or fixed furniture. "
                "Only respond with a call of this function, no other text."
            ),
            'parameters': {
                'type': 'OBJECT',
                'properties': {
                    'numbers_to_remove': {
                        'type': 'array',
                        'description': (
                            "List of identifiers for permanent architectural elements "
                            "(walls, built-ins, etc.) that cannot be removed without "
                            "construction work. These will be removed from furniture detection."
                        ),
                        'items': {
                            'type': 'string',
                        },
                    },
                    'explanation': {
                        'type_': 'STRING',
                        'description': (
                            "Detailed explanation of why each identified element is considered "
                            "a permanent architectural feature, referencing specific structural "
                            "or construction aspects."
                        )
                    }
                },
                'required': ['numbers_to_remove', 'explanation']
            }
        }]
    }

    # Convert the labeled PIL image to base64
    labeled_image = labeled_image.resize((512, 512))  # clamp size for simplicity
    buf = io.BytesIO()
    labeled_image.save(buf, format="JPEG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    parts = [
        {'mime_type': 'image/jpeg', 'data': b64_image},
        prompt
    ]

    try:
        from google.generativeai import GenerativeModel
        model_instance = GenerativeModel(model_name=model, tools=architectural_tool)
        chat = model_instance.start_chat()
        response = chat.send_message(parts)
        print("[Architectural] Response from Gemini:\n", response)
        
        # Look through the content for a tool use
        for candidate in response.candidates:
            if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts'):
                continue
            for part in candidate.content.parts:
                print(part)
                if not hasattr(part, 'function_call'):
                    continue
                fn_call = part.function_call
                if not hasattr(fn_call, 'args'):
                    continue
                args_dict = fn_call.args
                # parse
                letters = list(args_dict.get('letters_to_remove', []))
                explanation = str(args_dict.get('explanation', ''))
                return (letters, explanation)

        print("[Architectural] No valid function call found in response.")
        return ([], "No valid function call found in Gemini response.")

    except Exception as e:
        print("[Architectural] Error calling Gemini:", e)
        raise

    return ([], "Unexpected error occurred.")

if __name__ == "__main__":
    local_img = Image.open("test.png")
    local_img = local_img.resize((512, 512))
    letters, explanation = analyze_architectural_elements(local_img)
    print(letters, explanation)