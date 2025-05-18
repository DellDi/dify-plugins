## Dify 插件开发指南

你好，看起来你已经创建了一个插件，现在让我们开始开发吧！

### 选择你想要开发的插件类型

在开始之前，你需要了解一些关于插件类型的基本知识。Dify 插件支持扩展以下能力：
- **工具(Tool)**: 工具提供商，如谷歌搜索、Stable Diffusion 等，可用于执行特定任务。
- **模型(Model)**: 模型提供商，如 OpenAI、Anthropic 等，你可以使用它们的模型来增强 AI 能力。
- **端点(Endpoint)**: 类似于 Dify 中的服务 API 和 Kubernetes 中的 Ingress，你可以将 HTTP 服务扩展为端点，并使用自己的代码控制其逻辑。

根据你想要扩展的能力，我们将插件分为三种类型：**工具(Tool)**、**模型(Model)** 和 **扩展(Extension)**。

- **工具(Tool)**: 它是一个工具提供者，但不仅限于工具，你还可以在其中实现端点。例如，如果你要构建 Discord 机器人，则需要同时实现“发送消息”和“接收消息”功能，这时同时需要**工具**和**端点**。
- **模型(Model)**: 仅作为模型提供者，不允许扩展其他功能。
- **扩展(Extension)**: 有时，你可能只需要一个简单的 HTTP 服务来扩展功能，这时**扩展**是正确的选择。

我们当前的“实体查找器”插件选择了**工具(Tool)**类型，非常适合我们的需求。如果需要更改，可以通过修改 `manifest.yaml` 文件来实现。

### Manifest 文件详解

现在你可以编辑 `manifest.yaml` 文件来描述你的插件，下面是它的基本结构：

- **version** (版本, 必需)：插件的版本号。
- **type** (类型, 必需)：插件的类型，目前只支持 `plugin`，未来会支持 `bundle`。
- **author** (作者, 必需)：作者，它是插件市场中的组织名称，也应与仓库所有者名称一致。
- **label** (标签, 必需)：多语言名称。
- **created_at** (创建时间, 必需)：创建时间，插件市场要求创建时间必须早于当前时间。
- **icon** (图标, 必需)：图标路径。
- **resource** (资源, 对象)：要应用的资源。
  - **memory** (内存, `int64`)：最大内存使用量，主要与 Serverless SaaS 上的资源申请相关，单位为字节。
  - **permission** (权限, 对象)：权限申请。
    - **tool** (工具, 对象)：反向调用工具权限。
      - **enabled** (是否启用, `bool`)
    - **model** (模型, 对象)：反向调用模型权限。
      - **enabled** (是否启用, `bool`)
      - **llm** (是否启用 LLM, `bool`)
      - **text_embedding** (是否启用文本嵌入, `bool`)
      - **rerank** (是否启用重排, `bool`)
      - **tts** (是否启用 TTS, `bool`)
      - **speech2text** (是否启用语音转文本, `bool`)
      - **moderation** (是否启用内容审核, `bool`)
    - **node** (节点, 对象)：反向调用节点权限。
      - **enabled** (是否启用, `bool`)
    - **endpoint** (端点, 对象)：允许注册端点权限。
      - **enabled** (是否启用, `bool`)
    - **app** (应用, 对象)：反向调用应用权限。
      - **enabled** (是否启用, `bool`)
    - **storage** (存储, 对象)：申请持久化存储权限。
      - **enabled** (是否启用, `bool`)
      - **size** (大小, `int64`)：允许的最大持久化内存，单位为字节。
- **plugins** (插件扩展, 必需)：插件扩展能力的 YAML 文件列表，插件包内的绝对路径。如果需要扩展模型，你需要定义一个类似 `openai.yaml` 的文件，并在此处填写路径，该路径下的文件必须存在，否则打包会失败。
  - **格式**
    - **tools** (`list[string]`): 扩展的工具提供者，详细格式请参考 [工具指南](https://docs.dify.ai/plugins/schema-definition/tool)
    - **models** (`list[string]`)：扩展的模型提供者，详细格式请参考 [模型指南](https://docs.dify.ai/plugins/schema-definition/model)
    - **endpoints** (`list[string]`)：扩展的端点提供者，详细格式请参考 [端点指南](https://docs.dify.ai/plugins/schema-definition/endpoint)
  - **限制**
    - 不允许同时扩展工具和模型
    - 不允许没有任何扩展
    - 不允许同时扩展模型和端点
    - 目前每种类型的扩展只支持一个供应商
- **meta** (元数据, 对象)
  - **version** (版本, 必需)：manifest 格式版本，初始版本为 `0.0.1`
  - **arch** (架构列表, 必需)：支持的架构，目前只支持 `amd64` 和 `arm64`
  - **runner** (运行时配置, 必需)：运行时配置
    - **language** (语言, 字符串)：目前只支持 `python`
    - **version** (版本, 字符串)：语言版本，目前只支持 `3.12`
    - **entrypoint** (入口点, 字符串)：程序入口，在 Python 中应为 `main`

### 安装依赖

- 首先，你需要一个 Python 3.11+ 环境，因为我们的 SDK 需要它。
- 然后，安装依赖：
    ```bash
    pip install -r requirements.txt
    ```
- 如果你想添加更多依赖，可以将它们添加到 `requirements.txt` 文件中。一旦你在 `manifest.yaml` 文件中将运行时设置为 Python，`requirements.txt` 将会自动生成并用于打包和部署。

### 实现插件

现在你可以开始实现你的插件了，通过以下示例，你可以快速了解如何实现自己的插件：

- [OpenAI](https://github.com/langgenius/dify-plugin-sdks/tree/main/python/examples/openai)：模型提供商的最佳实践
- [Google Search](https://github.com/langgenius/dify-plugin-sdks/tree/main/python/examples/google)：工具提供商的简单示例
- [Neko](https://github.com/langgenius/dify-plugin-sdks/tree/main/python/examples/neko)：端点组的有趣示例

### 测试和调试插件

你可能已经注意到你的插件根目录中有一个 `.env.example` 文件，只需将其复制为 `.env` 并填写相应的值。如果你想在本地调试你的插件，需要设置一些环境变量。

- `INSTALL_METHOD`: 将其设置为 `remote`，你的插件将通过网络连接到 Dify 实例。
- `REMOTE_INSTALL_HOST`: 你的 Dify 实例的主机地址，你可以使用我们的 SaaS 实例 `https://debug.dify.ai`，或自托管的 Dify 实例。
- `REMOTE_INSTALL_PORT`: 你的 Dify 实例的端口，默认为 5003。
- `REMOTE_INSTALL_KEY`: 你应该从你使用的 Dify 实例获取调试密钥，在插件管理页面的右上角，你会看到一个带有 `debug` 图标的按钮，点击它即可获取密钥。

运行以下命令启动你的插件：

```bash
python -m main