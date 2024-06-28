from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "rbgo/inferless-llama-3-8B"  # Specify the model repository ID
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)
        self.llm = LLM(model=model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    def infer(self,inputs):
        prompts = inputs["prompt"]  # Extract the prompt from the input
        chat_format = [{"role": "user", "content": prompts}]
        text = self.tokenizer.apply_chat_template(chat_format,tokenize=False,add_generation_prompt=True)
        result = self.llm.generate(text, self.sampling_params)
        return {'generated_text': output.outputs[0].text}
