# 导入所需的库
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

"""
Cohere重排算法实现

Cohere Rerank是由Cohere公司提供的商业级重排API服务，基于先进的语言模型技术。

核心特点：
1. 企业级性能：基于大规模预训练模型，具备强大的语义理解能力
2. 多语言支持：支持包括中文在内的多种语言的重排任务
3. 即用即得：无需本地部署模型，通过API调用即可使用
4. 持续优化：模型持续更新，性能不断提升

技术优势：
- 高精度：基于先进的Transformer架构和大规模训练数据
- 低延迟：优化的推理引擎，支持实时重排需求
- 易集成：标准的REST API接口，易于集成到现有系统
- 可扩展：支持大批量文档的并行重排处理

适用场景：
- 商业级搜索系统
- 对精度要求较高的应用
- 快速原型开发和测试
- 多语言检索系统

成本考虑：
- 按API调用次数计费
- 适合中小规模应用或对成本不敏感的场景
- 建议在开发阶段进行成本评估
"""

print("🔄 初始化Cohere重排服务...")

# 1. API密钥配置
print("🔐 配置Cohere API密钥...")
print("📝 获取API密钥地址：https://dashboard.cohere.com/api-keys")

# 获取Cohere API key - 两种配置方式
import os

# 方式1：从环境变量读取（推荐）
api_key_from_env = os.getenv('CO_API_KEY')

# 方式2：直接设置（仅用于测试，生产环境请使用环境变量）
api_key = 'XXXX'  # 请替换为您的实际API密钥
os.environ['CO_API_KEY'] = api_key

if api_key_from_env:
    print("✅ 从环境变量成功读取API密钥")
else:
    print("⚠️  使用硬编码API密钥（请在生产环境中使用环境变量）")

print("🔒 安全提醒：请确保API密钥的安全，不要将其提交到代码仓库")

# 2. 准备示例文档
print("\n📋 准备测试文档...")
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
