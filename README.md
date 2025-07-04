# AI Document Assistant

一个基于FastAPI和LangChain的智能文档助手，帮助技术人员高效查阅和检索技术文档。

## 功能特点

- 📄 支持多种文档格式（PDF, Markdown, txt等）
- 🔍 基于语义的文档检索
- ❓ 智能问答功能
- 🚀 高性能的API接口
- 🛠️ 易于扩展和维护的模块化设计

## 技术栈

- **后端框架**: FastAPI
- **AI/ML**: LangChain, Hugging Face, PyTorch
- **向量存储**: Chroma
- **数据库**: SQLite (默认)
- **认证**: JWT

## 快速开始

1. 克隆仓库
   ```bash
   git clone <repository-url>
   cd ai-document-assistant
   ```
如果本地配置有多个github用户，可以:
```bash
git remote set-url origin git@github.com-chaneasonduo:chaneasonduo/aidoc-task.git
```

2. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   .\venv\Scripts\activate  # Windows
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量
   复制 `.env.example` 到 `.env` 并配置相关参数

5. 启动服务
   ```bash
   uvicorn app.main:app --reload
   ```

6. 访问API文档
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 项目结构

```
ai-document-assistant/
├── app/                    # 应用代码
│   ├── api/                # API路由
│   ├── core/               # 核心配置
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   └── utils/              # 工具函数
├── data/                   # 数据文件
├── static/                 # 静态文件
├── tests/                  # 测试代码
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## API文档

启动服务后，可以访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 开发

### 代码规范

- 使用 `black` 进行代码格式化
- 使用 `isort` 进行导入排序
- 使用 `mypy` 进行类型检查

### 测试

运行测试：

```bash
pytest
```

## 部署

### 使用Docker

```bash
docker build -t ai-document-assistant .
docker run -d -p 8000:8000 ai-document-assistant
```

## dashscope api
```curl
curl --location "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \
--header "Authorization: Bearer $DASHSCOPE_API_KEY" \
--header "Content-Type: application/json" \
--data '{
    "model": "qwen-plus",
    "input":{
        "messages":[      
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "你是谁？"
            }
        ]
    },
    "parameters": {
        "result_format": "message"
    }
}'
```

## 贡献

欢迎提交Issue和PR。

## 许可证

MIT
