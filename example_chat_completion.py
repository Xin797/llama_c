# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "how to lead a team on a project?"}],
        [
            {"role": "user", "content": "What are some famous landmarks in Japan?"},
            {
                "role": "assistant",
                "content": """\
    Japan is home to many iconic landmarks that attract tourists from all over the world. Here are some famous landmarks in Japan:

    1. Mount Fuji: This majestic and iconic mountain is the highest peak in Japan and a symbol of the country.
    2. Tokyo Tower: A prominent landmark in Tokyo, this communications and observation tower offers panoramic views of the city.
    3. Kyoto's Kiyomizu-dera Temple: A historic temple with a stunning wooden stage that offers picturesque views of Kyoto.

    These are just a few examples of the many famous landmarks that Japan has to offer.""",
            },
            {"role": "user", "content": "What are the benefits of practicing yoga regularly?"},
        ],
        [
            {"role": "system", "content": "How can I improve my time management skills at work?"},
        ],
        [
            {"role": "user", "content": "Can you recommend some must-read books for entrepreneurs?"},
            {
                "role": "assistant",
                "content": """\
    Reading is a great way to enhance your entrepreneurial skills and knowledge. Here are some must-read books for entrepreneurs:

    1. "The Lean Startup" by Eric Ries: This book introduces the concept of lean methodology and is essential for any startup founder.
    2. "The Innovator's Dilemma" by Clayton Christensen: A classic that explores disruptive innovation and its impact on businesses.
    3. "Shoe Dog" by Phil Knight: The memoir of the Nike founder offers valuable insights into building a successful business.

    These books cover a range of important topics for entrepreneurs and are highly recommended reads.""",
            },
            {"role": "user", "content": "What are the best ways to stay productive while working from home?"},
        ],
        [
            {"role": "user", "content": "What are some traditional customs and festivals in India?"},
            {
                "role": "assistant",
                "content": """\
    India is known for its rich cultural heritage and vibrant festivals. Here are some traditional customs and festivals in India:

    1. Diwali: Also known as the Festival of Lights, Diwali is one of the most important Hindu festivals celebrated with fireworks, sweets, and decorations.
    2. Holi: Known as the Festival of Colors, Holi is a lively celebration where people throw colored powders and water at each other in a joyous atmosphere.
    3. Navratri: A nine-night festival dedicated to the worship of the Hindu goddess Durga, Navratri features traditional dance performances and music.

    These are just a few examples of the colorful customs and festivals that showcase India's cultural diversity.""",
            },
            {"role": "user", "content": "What are some key tips for taking great photographs?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)