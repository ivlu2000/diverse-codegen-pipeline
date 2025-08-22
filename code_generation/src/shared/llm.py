import asyncio
import openai
from tqdm import tqdm
from typing import List


class LLMClient:
    def __init__(
        self,
        base_url="http://localhost:8080/v1",
        api_key="no-key-needed",
        model="llama3.1-70b-8192",
    ):
        """
        Initialize the LLM client with the given base URL and API key.

        Args:
            base_url: URL of the load balancer
            api_key: API key (can be any non-empty string for local deployment)
        """
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def batched_inference(
        self,
        prompts: List[str],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        num_concurrent: int = 50,
    ) -> List[str]:
        """
        Create batches of size num_concurrent where each batch is sent concurrently to the server.

        Args:
            prompts: List of prompts to send to the LLM
            system_message: System message to include with each prompt
            temperature: Temperature parameter for generation
            num_concurrent: Number of concurrent requests to make

        Returns:
            List of generated responses
        """
        results = []
        for i in tqdm(range(0, len(prompts), num_concurrent), desc="Generating..."):
            batch_to_generate = [
                prompts[k] for k in range(i, min(i + num_concurrent, len(prompts)))
            ]
            batch_generated = asyncio.run(
                self.generate_batch(batch_to_generate, system_message, temperature)
            )

            results += batch_generated

            # Optional: Store intermediate results to disk if generating a lot of data

        return results

    async def generate_batch(
        self, batch_to_generate: List[str], system_message: str, temperature: float
    ) -> List[str]:
        """
        Concurrently send the prompts in the batch to the server using the openai chat endpoint.

        Args:
            batch_to_generate: List of prompts to generate
            system_message: System message to include with each prompt
            temperature: Temperature parameter for generation

        Returns:
            List of generated responses
        """
        batch_generated = await asyncio.gather(
            *[
                self.openai_chat_endpoint(prompt, system_message, temperature)
                for prompt in batch_to_generate
            ]
        )

        return batch_generated

    async def openai_chat_endpoint(
        self, prompt: str, system_message: str, temperature: float
    ) -> str:
        """
        Standard usage of openai client chat completions endpoint.

        Args:
            prompt: The prompt to send to the LLM
            system_message: System message to include with the prompt
            temperature: Temperature parameter for generation

        Returns:
            Generated response
        """
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=temperature,
                top_p=0.95,
                timeout=1000000,  # Important! If too low, request can time out in server queue
            )

            result = chat_completion.choices[0].message.content

        # You can also implement a retry mechanism here, although it is unlikely that something first goes wrong and then works in a second try.
        except Exception as e:
            print(str(e))
            result = "Error occured: " + str(e)

        # Optional: Print intermediate results for checking
        print(result)
        print("######################################")
        return result


if __name__ == "__main__":
    # Example usage:
    en_text = """
Germany,[e] officially the Federal Republic of Germany (FRG),[f] is a country in Central Europe.
It lies between the Baltic and North Sea to the north and the Alps to the south.
"""

    llm_client = LLMClient()
    prompts = [en_text for i in range(200)]
    system_message = (
        "You are a helpful assistant that translates text from English to German."
    )
    results = llm_client.batched_inference(
        prompts, system_message, temperature=0.0, num_concurrent=50
    )
