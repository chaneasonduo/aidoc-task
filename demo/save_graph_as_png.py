# 将 LangGraph 的 graph 保存为 PNG 图片的示例代码

# 导入必要的库
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, TypedDict
import os

# 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 创建一个简单的示例图用于演示
# 实际应用中，您可以使用自己已经构建好的 graph

def create_sample_graph():
    # 创建图构建器
    graph_builder = StateGraph(State)
    
    # 定义示例节点
    def node1(state: State):
        # 这里是节点1的实际逻辑
        return {"messages": ["节点1执行完毕"]}
    
    def node2(state: State):
        # 这里是节点2的实际逻辑
        return {"messages": ["节点2执行完毕"]}
    
    # 添加节点
    graph_builder.add_node("node1", node1)
    graph_builder.add_node("node2", node2)
    
    # 添加边
    graph_builder.add_edge(START, "node1")
    graph_builder.add_edge("node1", "node2")
    graph_builder.add_edge("node2", END)
    
    # 编译图
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph

# 方法1: 直接保存 draw_mermaid_png() 的结果
# 这是最直接的方法，但需要确保已安装必要的依赖

def save_graph_directly(graph, output_path="workflow.png"):
    """
    直接将 graph 保存为 PNG 图片
    
    参数:
    graph: 已编译的 LangGraph 图对象
    output_path: 输出文件路径，默认为 "workflow.png"
    """
    try:
        # 获取 graph 的可视化表示并保存为 PNG
        png_data = graph.get_graph().draw_mermaid_png()
        
        # 将二进制数据写入文件
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"图已成功保存为: {output_path}")
        return True
    except Exception as e:
        print(f"保存图时出错: {e}")
        print("提示: 可能需要安装额外的依赖，如 mermaid-cli")
        return False

# 方法2: 先获取 mermaid 代码，然后保存 (需要外部工具处理)

def save_mermaid_code(graph, output_path="workflow.mmd"):
    """
    保存 graph 的 mermaid 代码到文件
    之后可以使用 mermaid-cli 等工具将其转换为 PNG
    
    参数:
    graph: 已编译的 LangGraph 图对象
    output_path: mermaid 代码输出文件路径
    """
    try:
        # 获取 graph 的 mermaid 代码
        mermaid_code = graph.get_graph().draw_mermaid()
        
        # 将代码写入文件
        with open(output_path, "w") as f:
            f.write(mermaid_code)
        
        print(f"Mermaid 代码已保存为: {output_path}")
        print("提示: 可以使用 mermaid-cli 将其转换为 PNG: ")
        print(f"       mmdc -i {output_path} -o {output_path.replace('.mmd', '.png')}")
        return True
    except Exception as e:
        print(f"保存 mermaid 代码时出错: {e}")
        return False

# 主函数
if __name__ == "__main__":
    # 创建示例图
    print("创建示例图...")
    graph = create_sample_graph()
    
    # 方法1: 直接保存为 PNG
    print("\n尝试直接保存为 PNG...")
    success = save_graph_directly(graph)
    
    # 如果方法1失败，使用方法2
    if not success:
        print("\n尝试保存 mermaid 代码...")
        save_mermaid_code(graph)
    
    # 附加信息
    print("\n依赖安装提示:")
    print("1. 确保已安装 langgraph: pip install langgraph")
    print("2. 对于直接保存 PNG 功能，可能需要安装额外依赖")
    print("3. 对于 mermaid-cli 方法，请安装: npm install -g @mermaid-js/mermaid-cli")