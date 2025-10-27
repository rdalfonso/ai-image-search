from pydoc import doc
from openai import OpenAI
import os
import json
import base64
import chromadb
from agents import Agent, Runner, trace
import gradio as gr
from pydantic import BaseModel, Field, ValidationError

class Me:

    def __init__(self):
        self.query = "cat"
        self.clientChroma = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.clientChroma.get_or_create_collection(name="images-description-collection")

    def system_prompt(self):
        system_prompt = f"Want to search the documents in the chromadb database. The collection has image descriptions."
        print(system_prompt)
        return system_prompt
    
    def chat(self, message, history):
        print('message:', message)
        results = self.collection.query(
            query_texts=[message],
            n_results=5
        )

        images = []

        # FIRST: See what distances you're actually getting
        print("\n=== All Results with Distances ===")
        for i, (distance, doc) in enumerate(zip(results['distances'][0], results['documents'][0])):
            print(f"Distance: {distance:.4f} - Doc: {doc[:80]}...")

        # NOW filter with a realistic threshold
        # For ChromaDB with default settings, distances can be 1.0+
        # You want to filter out results that are TOO FAR, not too close
        DISTANCE_THRESHOLD = 1.2  # Try 1.2, 1.5, or even 2.0 depending on your output above

        filtered_docs = []
        for i, distance in enumerate(results['distances'][0]):
            if distance < DISTANCE_THRESHOLD:  # Keep results UNDER this threshold
                filtered_docs.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'distance': distance
                })

        print(f"\n=== Filtered Results (distance < {DISTANCE_THRESHOLD}) ===")
        print(f"Found {len(filtered_docs)} relevant documents")
        for doc in filtered_docs:
            image_name = doc['id'].replace('emb-', '') + "_rn.jpg"
            image_path = os.path.join("./images_renamed", image_name)
            print(f"ID: {image_path}")
            print(f"Distance: {doc['distance']:.4f} - {doc['document'][:80]}...")
            caption = f"{doc['document'][:80]}"
            images.append((image_path, caption))

        status = f"Found {len(images)} images" if images else "Sorry, I couldn't find any relevant documents."
        return images, status  # Return multiple outputs


# Pydantic model for structured response
class ImageDescription(BaseModel):
    description: str = Field(..., min_length=1, description="A short description of the image")
    name: str = Field(..., min_length=1, max_length=50, description="A few words for filename (no spaces)")
    
    def get_safe_filename(self) -> str:
        """Generate a safe filename from the name field"""
        import re
        # Remove special characters and normalize
        safe_name = re.sub(r'[^\w\s-]', '', self.name).strip()
        return f"{safe_name.replace(' ', '_').lower()}_rn.jpg"

# Must match LM Studio model name
model = "google/gemma-3n-e4b"  

# Connect to LM Studio's local API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Directory with your images
images_dir = "./images"

# Create a directory for renamed images
renamed_dir = "./images_renamed"
os.makedirs(renamed_dir, exist_ok=True)

# Initialize the Chroma client
clientChroma = chromadb.PersistentClient(path="./chroma_db")
collection = clientChroma.get_or_create_collection(name="images-description-collection")
documents_data = []

# Generate JSON schema from Pydantic model
def get_json_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "image_description",
            "schema": ImageDescription.model_json_schema()
        }
    }


# Loop over all supported image files
for filename in os.listdir(images_dir):
    if filename.lower().endswith(".jpg"):
        print(f"filename: {filename}")

        # Path to your image
        image_path = os.path.join(images_dir, filename)
        print(f"image_path: {image_path}")

        # Read and base64-encode the image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

            response = client.chat.completions.create(
                model=model,
                messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": (
                                "What's on this image? Return JSON with two fields: "
                                "`description` (a short description), and "
                                "`name` (a few words with no spaces)."
                                )},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                response_format=get_json_schema()
               
        )


        # handle json structured output response.
        raw = response.choices[0].message.content
        try:

             # Parse the response into our Pydantic model
            image_info = ImageDescription.model_validate_json(raw)

            # Set values from response
            description = image_info.description
            name = image_info.name

            # Always use .jpg for new name
            # Generate safe filename
            new_filename = image_info.get_safe_filename()
            new_path = os.path.join(images_dir, new_filename)
            os.rename(image_path, new_path)

            os.makedirs(renamed_dir, exist_ok=True)
            final_path = os.path.join(renamed_dir, new_filename)
            os.rename(new_path, final_path)

            # Print result
            embeddingID = f"emb-{name.lower().replace(' ', '_')}" 
            print(f"Embedding ID: {embeddingID}\n")
            print('-------------------')


            # Create documents_data array
            documents_data.append({
                "id": embeddingID,
                "text": description
            })
         
        except json.JSONDecodeError:
            print(f"Could not parse JSON: {raw}")


 # Add documents to the collection and query it
if documents_data:
  for doc in documents_data:
    collection.add(
      documents=[doc["text"]],
      ids=[doc["id"]]            
    )


if __name__ == "__main__":
    me = Me()
    with gr.Blocks() as demo:
        search_box = gr.Textbox(label="Search")
        status = gr.Textbox(label="Status")
        gallery = gr.Gallery(label="Results", columns=3)
        
        # Map outputs to components
        search_box.submit(
            me.chat, 
            inputs=search_box, 
            outputs=[gallery, status]
        )

demo.launch()