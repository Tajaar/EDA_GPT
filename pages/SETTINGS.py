import streamlit as st

class Settings:
    def __init__(self):
        """Load API keys from Streamlit secrets"""
        self.api_keys = {
            "HuggingFace Hub API Token": st.secrets.get("HUGGINGFACEHUB_API_TOKEN", ""),
            "Google Gemini API Key": st.secrets.get("GOOGLE_GEMINI_API", ""),
            "OpenAI API Key": st.secrets.get("OPENAI_API_KEY", ""),
            "Groq API Key": st.secrets.get("GROQ_API_KEY", ""),
            "Anthropic API Key": st.secrets.get("ANTHROPIC_API_KEY", "")
        }
        self.set_session_state_defaults()

    def set_session_state_defaults(self):
        """Ensure session state variables exist"""
        for key, value in self.api_keys.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def changesettings(self):
        """Display API Key settings and model selection UI"""
        st.title("⚙️ Settings")

        with st.expander('🔑 **Provide API Keys**', expanded=True):
            for key in self.api_keys.keys():
                st.session_state[key] = st.text_input(key, st.session_state[key], type="password")

            if st.button('✅ Apply API Keys'):
                st.success("✅ API keys updated successfully!")

        with st.expander('🎯 **Choose Model**', expanded=True):
            model_choices = {
                "HuggingFace - Starcoder": "huggingface/starcoder",
                "OpenAI - GPT-4": "gpt-4",
                "Anthropic - Claude-2": "anthropic/claude-2",
                "Google Gemini - Pro": "google/gemini-pro",
                "Google Gemini - Pro Vision": "google/gemini-pro-vision"
            }

            selected_model = st.radio("Choose a Model:", list(model_choices.keys()), index=0)

            if st.button("✅ Apply Model"):
                st.session_state["selected_model"] = model_choices[selected_model]
                st.success(f"✅ Model set to: {selected_model}")

# ✅ Main function to run the app
def main():
    settings = Settings()
    settings.changesettings()

# ✅ Run the UI
if __name__ == "__main__":
    main()
