# Import required libraries
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

"""
Cohere Reranking Algorithm Implementation

Cohere Rerank is a commercial-grade reranking API service provided by Cohere, built on advanced language model technology.

Core features:
1. Enterprise-grade performance: based on large-scale pretrained models with strong semantic understanding
2. Multilingual support: handles reranking tasks across many languages, including Chinese
3. Ready to use: no need to deploy models locally, just call the API
4. Continuous optimization: models are updated continuously, with performance improving over time

Technical advantages:
- High precision: based on advanced Transformer architecture and large-scale training data
- Low latency: optimized inference engine supporting real-time reranking needs
- Easy integration: standard REST API interface, easy to integrate into existing systems
- Scalable: supports parallel reranking of large batches of documents

Suitable scenarios:
- Commercial-grade search systems
- Applications with high precision requirements
- Rapid prototyping and testing
- Multilingual retrieval systems

Cost considerations:
- Billed per API call
- Suitable for small-to-medium scale applications or cost-insensitive scenarios
- Recommended to evaluate cost during the development phase
"""

print("🔄 Initializing Cohere reranking service...")

# 1. API key configuration
print("🔐 Configuring Cohere API key...")
print("📝 Get your API key here: https://dashboard.cohere.com/api-keys")

# Get the Cohere API key - two configuration methods
import os

# Method 1: read from environment variable (recommended)
api_key_from_env = os.getenv('CO_API_KEY')

# Method 2: set directly (for testing only; use environment variables in production)
api_key = 'XXXX'  # Replace with your actual API key
os.environ['CO_API_KEY'] = api_key

if api_key_from_env:
    print("✅ Successfully read API key from environment variable")
else:
    print("⚠️  Using hardcoded API key (use an environment variable in production)")

print("🔒 Security reminder: keep your API key safe and never commit it to a code repository")

# 2. Prepare sample documents
print("\n📋 Preparing test documents...")
documents = [
    Document(
        page_content="Mount Wutai是中国四大佛教名山之一，以文殊菩萨道场闻名。",
        metadata={"source": "山西旅游指南", "category": "佛教文化", "location": "忻州市"}
    ),
    Document(
        page_content="Yungang Grottoes是中国三大石窟之一，以精美的佛教雕塑著称。",
        metadata={"source": "山西旅游指南", "category": "石窟艺术", "location": "大同市"}
    ),
    Document(
        page_content="Pingyao Ancient City是中国保存最完整的古代县城之一，被列为世界文化遗产。",
        metadata={"source": "山西旅游指南", "category": "古建筑", "location": "晋中市"}
    )
]

print(f"文档数量: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  文档 {i}:")
    print(f"    内容: {doc.page_content}")
    print(f"    来源: {doc.metadata.get('source', '未知')}")
    print(f"    类别: {doc.metadata.get('category', '未知')}")
    print(f"    位置: {doc.metadata.get('location', '未知')}")

# 3. 创建BM25检索器（作为初始检索）
print(f"\n🔍 创建BM25初始检索器...")
print("  BM25用于第一阶段检索，提供候选文档集合")
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # 设置返回前3个结果
print(f"✅ BM25检索器配置完成，返回Top-{retriever.k}结果")

# 4. 设置Cohere重排序器
print(f"\n🤖 配置Cohere重排序器...")
reranker = CohereRerank(
    model="rerank-multilingual-v3.0"  # 多语言重排模型，支持中文
)
print(f"使用模型: rerank-multilingual-v3.0")
print("  模型特点:")
print("  - ✅ 支持多语言（包括中文）")
print("  - ✅ 基于先进的Transformer架构")
print("  - ✅ 针对重排任务专门优化")
print("  - ✅ 持续更新和改进")

# 5. 执行查询和重排
print(f"\n🎯 开始执行查询和重排...")
query = "山西有哪些著名的旅游景点？"
print(f"查询: {query}")

print(f"\n第一阶段 - BM25初始检索:")
print("  🔍 使用BM25算法进行初始检索...")
initial_docs = retriever.invoke(query)
print(f"  📊 BM25检索到 {len(initial_docs)} 个候选文档")

for i, doc in enumerate(initial_docs, 1):
    print(f"    {i}. {doc.page_content}")

print(f"\n第二阶段 - Cohere重排:")
print("  🤖 调用Cohere API进行语义重排...")
print("  ⏳ 正在处理中（可能需要几秒钟）...")

try:
    # 使用Cohere重排序器对BM25结果进行重排
    reranked_docs = reranker.compress_documents(
        documents=initial_docs, 
        query=query
    )
    print("  ✅ Cohere重排完成")
    
    # 6. 输出重排结果
    print(f"\n{'='*60}")
    print(f"🏆 Cohere重排最终结果")
    print(f"{'='*60}")
    print(f"查询: {query}")
    print(f"\n重排序后的结果（按相关性降序）:")
    
    for i, doc in enumerate(reranked_docs, 1):
        print(f"\n📄 排名 {i}:")
        print(f"   文档内容: {doc.page_content}")
        
        # 显示文档元数据
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"   文档来源: {doc.metadata.get('source', '未知')}")
            print(f"   景点类别: {doc.metadata.get('category', '未知')}")
            print(f"   所在位置: {doc.metadata.get('location', '未知')}")
        
        # 如果有重排分数，显示出来
        if hasattr(doc, 'score'):
            print(f"   重排分数: {doc.score:.4f}")

except Exception as e:
    print(f"  ❌ Cohere API调用失败: {str(e)}")
    print("  💡 可能的原因:")
    print("    - API密钥无效或已过期")
    print("    - 网络连接问题")
    print("    - API配额已用完")
    print("    - 请检查API密钥配置和网络状态")

print(f"\n📋 Cohere重排总结:")
print("- ✅ 企业级性能：基于大规模预训练模型")
print("- ✅ 多语言支持：原生支持中文等多种语言")
print("- ✅ 即用即得：无需本地部署，API调用即可使用")
print("- ✅ 持续优化：模型定期更新，性能不断提升")
print("- 💰 成本考虑：按API调用次数计费")
print("- 🔐 安全提醒：保护好您的API密钥")
print("- 📈 适用场景：商业级搜索、快速原型、多语言应用")
