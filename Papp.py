import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the paraphrasing model and tokenizer
@st.cache_resource
def load_paraphrasing_model():
    """
    Load the paraphrasing model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
    model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
    return tokenizer, model

# Paraphrasing function
def paraphrase(
    question,
    tokenizer,
    model,
    num_beams=5,
    num_beam_groups=1,  # Ensuring a safe default
    diversity_penalty=0.0,  # Ensuring it aligns with num_beam_groups
    max_length=128,
):
    """
    Generate a single paraphrased version of the input text using the model.

    Args:
        question: The input text to paraphrase.
        tokenizer: The tokenizer for the paraphrasing model.
        model: The paraphrasing model.
        num_beams: Beam search width.
        num_beam_groups: Number of beam groups for diversity (default 1 for safety).
        diversity_penalty: Penalty for lack of diversity (default 0.0).
        max_length: Maximum length of paraphrased output.

    Returns:
        A single paraphrased sentence.
    """
    # Ensure valid configuration for group beam search
    if num_beam_groups > 1 and diversity_penalty <= 0.0:
        raise ValueError(
            "`diversity_penalty` must be > 0.0 when `num_beam_groups` > 1."
        )

    input_ids = tokenizer(
        f"paraphrase: {question}",
        return_tensors="pt",
        padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
        max_length=max_length,
        early_stopping=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Streamlit Application
def main():
    st.title("📚 Simple Paraphrasing Tool")
    st.markdown("This tool generates a single high-quality paraphrase for the given input text.")

    # Load model and tokenizer
    tokenizer, model = load_paraphrasing_model()

    # Text input box
    input_text = st.text_area("Enter Text:", placeholder="Type or paste your text here...", height=150)

    # Paraphrasing button
    if st.button("🔄 Paraphrase"):
        if not input_text.strip():
            st.warning("Please provide text to paraphrase.")
        else:
            with st.spinner("Generating paraphrase..."):
                try:
                    paraphrased_result = paraphrase(
                        input_text,
                        tokenizer=tokenizer,
                        model=model,
                        num_beams=5,
                        num_beam_groups=1,  # Single beam group
                        diversity_penalty=0.0,  # No diversity penalty
                    )
                    st.markdown("### Paraphrased Result:")
                    st.write(paraphrased_result)

                    # Option to download the paraphrased result
                    st.download_button(
                        label="📥 Download Paraphrased Result",
                        data=paraphrased_result,
                        file_name="paraphrased_result.txt",
                        mime="text/plain",
                    )
                except ValueError as ve:
                    st.error(f"Error during paraphrasing: {ve}")

if __name__ == "__main__":
    main()
