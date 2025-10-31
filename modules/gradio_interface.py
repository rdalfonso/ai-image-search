"""
Gradio interface for the image search application
"""

import gradio as gr

from .search_interface import ImageSearchInterface


def create_gradio_interface(search_interface: ImageSearchInterface) -> gr.Blocks:
    """Create and configure Gradio interface"""
    
    with gr.Blocks(title="Image Search") as demo:
        gr.Markdown("#Image Search with ChromaDB")
        gr.Markdown("Search for images using natural language descriptions")
        
        with gr.Row():
            search_box = gr.Textbox(
                label="Search Query",
                placeholder="e.g., 'cat with blue eyes', 'sunset over mountains'",
                lines=1
            )
        
        status = gr.Textbox(label="Status", interactive=False)
        
        gallery = gr.Gallery(
            label="Search Results",
            columns=3,
            height=600,
            object_fit="contain",
            show_label=True,
            preview=True
        )
        
        search_box.submit(
            search_interface.search,
            inputs=search_box,
            outputs=[gallery, status]
        )
    
    return demo