"""
LLMLingua Text Compression Example
LLMLingua is a tool for compressing long text, reducing the token count while preserving core information
Suitable for post-retrieval processing in RAG systems, lowering LLM input costs
"""

from llmlingua import PromptCompressor

# =============================================================================
# 1. Initialize the PromptCompressor
# =============================================================================

# Force the use of the CPU device to avoid CUDA errors
# model_name: specifies the base model used for compression; Llama-2-7b is used here
# device_map: device mapping, "cpu" means running on CPU to avoid GPU out-of-memory issues
llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    device_map="cpu"  # explicitly specify CPU usage
)

# =============================================================================
# 2. Text compression example
# =============================================================================

# Compress long text content
# context: the original text content to be compressed
# instruction: compression instruction, tells the model how to perform the compression
# question: a related question (optional), helps the model understand the compression focus
# target_token: target token count; the compressed text should be kept within this length
compressed_prompt = llm_lingua.compress_prompt(
    context="The Yungang Grottoes are located at the southern foot of Wuzhou Mountain, 17 kilometers west of Datong City in Shanxi Province in northern China. The grottoes are carved into the mountainside and stretch for 1 kilometer from east to west. There are 45 major caves, 252 niches of various sizes, and more than 51,000 stone carvings, making it one of the largest ancient grotto complexes in China. Together with the Mogao Caves in Dunhuang, the Longmen Grottoes in Luoyang, and the Maijishan Grottoes in Tianshui, it is known as one of China's four great grotto art treasuries. It was designated as one of the first national key cultural heritage sites by the State Council in 1961, was added to the UNESCO World Heritage List on December 14, 2001, and was rated as one of the first national 5A-level tourist attractions by the National Tourism Administration on May 8, 2007.……",
    instruction="Compress while preserving the main content",
    question="",
    target_token=10  # set up the target token count
)

# Output the compressed text
print("=== Compressed text ===")
print(compressed_prompt)
print(compressed_prompt['compressed_prompt'])
print(f"Compression rate: {compressed_prompt.get('rate', 'N/A')}")

# =============================================================================
# 3. JSON data compression example
# =============================================================================

# Define the JSON data to be compressed
json_data = {
    "id": 987654,
    "name": "Wukong",
    "biography": "Sun Wukong, whose Buddhist name is Pilgrim, is one of the main characters in the classic Chinese mythological novel Journey to the West. Sun Wukong was born from a magic stone that had existed since the creation of heaven and earth. He learned the seventy-two transformations of earthly demonic arts from his master Bodhi, and obtained the Ruyi Golden Cudgel from the Dragon King's palace as his weapon. After wreaking havoc in heaven, he was imprisoned beneath Five Elements Mountain by the Buddha and rendered unable to move. Five hundred years later, Tang Monk passed by Five Elements Mountain on his journey to the West to obtain Buddhist scriptures, removed the talisman, and freed Sun Wukong. Overcome with gratitude, and guided by Guanyin Bodhisattva, Sun Wukong became Tang Monk's disciple and accompanied him on the journey west to obtain the scriptures."
}

# JSON compression configuration
# rate: compression ratio, between 0 and 1; the smaller the value, the more aggressive the compression
# compress: whether to compress this field
# pair_remove: whether key-value pairs are allowed to be removed
# value_type: the type of the value (number/string, etc.)
json_config = {
    "id": {"rate": 1, "compress": False, "pair_remove": False, "value_type": "number"},      # ID is not compressed, kept as-is
    "name": {"rate": 0.7, "compress": False, "pair_remove": False, "value_type": "string"}, # name is lightly compressed
    "biography": {"rate": 0.3, "compress": True, "pair_remove": False, "value_type": "string"}  # biography is heavily compressed
}

# Perform JSON compression - with exception handling added
try:
    print("\n=== Original JSON data ===")
    import json
    print(json.dumps(json_data, ensure_ascii=False, indent=2))

    compressed_json = llm_lingua.compress_json(json_data, json_config)

    # Output the compressed JSON
    print("\n=== Compressed JSON ===")
    print(compressed_json['compressed_prompt'])

except Exception as e:
    print(f"\n=== Error during JSON compression ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("Possible solutions:")
    print("1. Simplify the text content in the JSON")
    print("2. Avoid using special characters and escape characters")
    print("3. Adjust the compression parameters")

    # Try a fallback approach: compress only the text content
    print("\n=== Using fallback text compression approach ===")
    try:
        backup_compressed = llm_lingua.compress_prompt(
            context=json_data["biography"],
            instruction="Compress the biography content while preserving key information",
            question="",
            target_token=30
        )
        print("Compressed biography:")
        print(backup_compressed['compressed_prompt'])
    except Exception as backup_e:
        print(f"Fallback approach also failed: {backup_e}")

# =============================================================================
# Usage notes
# =============================================================================
"""
Main use cases for the compressor:
1. Compress retrieved documents in a RAG system to reduce input tokens
2. Compress historical conversation records while preserving context coherence
3. Compress structured data (JSON) while retaining key fields

Parameter tuning recommendations:
- target_token: set according to the context limit of the downstream LLM
- rate: adjust according to the importance of the information; set a higher value for key information
- compress: numeric data usually doesn't need to be compressed
"""