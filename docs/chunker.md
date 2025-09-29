# 文本分块器 (Chunker) 详解

## 文件概述

`plag_tool/core/chunker.py` 实现了文本分块（chunking）功能，用于将长文本切分成多个重叠的小段，便于后续的向量化和相似度比较。这是抄袭检测系统的基础模块。

## 主要类

### 1. `TextChunk` 类

这是一个数据模型类，表示一个文本块：

```python
class TextChunk(BaseModel):
    text: str           # 文本内容
    start_pos: int      # 在原文中的起始位置
    end_pos: int        # 在原文中的结束位置
    doc_id: str         # 文档标识符
    chunk_index: int    # 块的索引号
    chunk_hash: str     # 文本的哈希值（用于缓存）
```

**关键方法**：
- `_compute_hash()`: 计算文本的SHA256哈希值（取前16位），用于缓存和去重
- `overlaps_with()`: 判断两个块是否重叠，支持容错距离参数

### 2. `TextChunker` 类

核心的文本分块器类，提供多种分块策略。

**初始化参数**：
- `chunk_size`: 每个块的大小（字符数）
- `overlap`: 相邻块之间的重叠字符数

## 核心算法

### 滑动窗口分块算法 (`chunk_text`)

```python
def chunk_text(self, text: str, doc_id: str) -> List[TextChunk]:
```

**工作原理**：
1. 使用固定大小的窗口（`chunk_size`）在文本上滑动
2. 每次滑动的步长是 `chunk_size - overlap`
3. 这样相邻的块之间会有 `overlap` 个字符的重叠

**算法示例**：
```
假设 chunk_size=10, overlap=3
文本: "这是一段测试文本用于演示分块算法的工作原理"

分块结果:
块1: "这是一段测试文本用于" (位置 0-10)
块2: "文本用于演示分块" (位置 7-15)
块3: "分块算法的工作原理" (位置 12-20)

重叠部分:
- 块1与块2: "文本用于"
- 块2与块3: "分块"
```

**代码逻辑**：
```python
while start < text_length:
    end = min(start + self.chunk_size, text_length)
    chunk_text = text[start:end]
    # 创建 TextChunk 对象
    start += self.stride  # stride = chunk_size - overlap
```

### 句子边界分块 (`chunk_with_sentences`)

```python
def chunk_with_sentences(self, text: str, doc_id: str) -> List[TextChunk]:
```

**优化特性**：
- 尽量在句子边界处分割，保持语义完整性
- 支持中英文句号：`['。', '！', '？', '；', '.', '!', '?', ';', '\n\n']`
- 在理想分割点附近50个字符范围内寻找句子结束符

**算法流程**：
1. 计算理想的结束位置（基于 `chunk_size`）
2. 在理想位置前后50字符范围内搜索句子结束符
3. 如果找到句子结束符，在其后分割
4. 如果没找到，按原始位置分割
5. 下一个块从当前块结束位置减去重叠量开始

**为什么需要句子边界分割**：
- 保持语义完整性，避免在句子中间切断
- 特别适合中文文本，因为中文语义单位更依赖句子结构
- 提高后续向量化的质量

### 合并小块 (`merge_small_chunks`)

```python
def merge_small_chunks(self, chunks: List[TextChunk], min_size: int = 100):
```

**用途**：
- 将过小的文本块与相邻块合并
- 避免产生太多碎片化的小块
- 提高后续处理效率

**合并策略**：
- 如果当前块小于最小尺寸且与下一块属于同一文档，则合并
- 合并时处理重叠部分，避免重复内容
- 保持合并后块的位置信息正确

## 关键参数说明

### 1. **chunk_size** (默认500字符)
- **用途**: 每个块的目标大小
- **建议值**:
  - 中文文本：500-1000字符
  - 英文文本：300-800字符
  - 技术文档：200-400字符
- **影响**:
  - 过大：可能包含多个主题，降低精确度
  - 过小：上下文不足，影响语义理解

### 2. **overlap** (默认100字符)
- **用途**: 相邻块的重叠部分
- **建议值**: chunk_size 的 20-30%
- **影响**:
  - 过大：冗余信息增加，计算成本上升
  - 过小：可能遗漏边界处的重要信息

### 3. **stride** (步长)
- **计算**: `stride = chunk_size - overlap`
- **作用**: 决定窗口每次移动的距离
- **示例**: chunk_size=500, overlap=100 → stride=400

## 为什么需要重叠设计？

重叠设计解决了以下问题：

### 1. **保持上下文连续性**
```
原文: "人工智能技术正在快速发展，深度学习算法取得重大突破"

无重叠分割:
块1: "人工智能技术正在快速"
块2: "发展，深度学习算法取得"
→ 语义被切断

有重叠分割:
块1: "人工智能技术正在快速发展"
块2: "快速发展，深度学习算法取得重大突破"
→ 保持了语义连续性
```

### 2. **避免边界抄袭遗漏**
- 如果抄袭内容正好跨越块边界，无重叠可能导致检测失败
- 重叠确保边界处的内容在多个块中都有体现

### 3. **提高检测鲁棒性**
- 文本的微小变动不会大幅影响分块结果
- 增加了抄袭片段被检测到的概率

## 实际应用场景

### 在抄袭检测系统中的作用：

1. **文档预处理阶段**
   ```python
   # 源文档分块
   source_chunks = chunker.chunk_with_sentences(source_text, "source")

   # 目标文档分块
   target_chunks = chunker.chunk_with_sentences(target_text, "target")
   ```

2. **向量化准备**
   - 每个文本块独立生成向量嵌入
   - 块的大小影响嵌入质量和计算效率

3. **相似度比较**
   - 源文档的每个块与目标文档的所有块进行相似度计算
   - 重叠设计确保不遗漏任何可疑片段

4. **结果聚合**
   - 相邻的相似块可以合并成更大的抄袭片段
   - 块的位置信息用于生成详细报告

## 使用示例

### 基本使用
```python
from plag_tool.core.chunker import TextChunker

# 创建分块器
chunker = TextChunker(chunk_size=500, overlap=100)

# 基本分块
chunks = chunker.chunk_text(text, "document_1")

# 句子边界分块（推荐）
chunks = chunker.chunk_with_sentences(text, "document_1")

# 合并小块
merged_chunks = chunker.merge_small_chunks(chunks, min_size=50)
```

### 参数调优示例
```python
# 适合短文本的配置
short_text_chunker = TextChunker(chunk_size=200, overlap=50)

# 适合长文档的配置
long_text_chunker = TextChunker(chunk_size=800, overlap=200)

# 高精度检测配置（重叠更多）
precision_chunker = TextChunker(chunk_size=400, overlap=150)
```

## 性能考虑

### 时间复杂度
- 基本分块：O(n)，其中n为文本长度
- 句子边界分块：O(n×w)，其中w为搜索窗口大小（通常很小）

### 空间复杂度
- 存储所有块：O(n×r)，其中r为重叠率
- 重叠率 = overlap / chunk_size

### 优化建议
1. **合理设置参数**：避免过度重叠造成的存储浪费
2. **使用哈希缓存**：相同文本块不重复处理
3. **批量处理**：多个文档可以并行分块

## 总结

文本分块器是抄袭检测系统的核心基础模块，通过智能的分割策略和重叠设计，确保了：

- **语义完整性**：句子边界分割保持语义单元完整
- **检测完整性**：重叠设计防止边界抄袭遗漏
- **计算效率**：合适的块大小平衡了精度和性能
- **扩展性**：支持多种分块策略，适应不同场景需求

正确理解和配置分块器参数，对整个抄袭检测系统的性能有决定性影响。