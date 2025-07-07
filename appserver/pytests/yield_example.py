def number_generator(n):
    """
    生成器函数示例：使用yield生成数字序列
    """
    for i in range(n):
        yield i

def fibonacci_generator(n):
    """
    生成斐波那契数列的生成器
    """
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

class ResourceManager:
    """
    使用yield的上下文管理器示例
    """
    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.is_open = False
    
    def __enter__(self):
        print(f"打开资源: {self.resource_name}")
        self.is_open = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"关闭资源: {self.resource_name}")
        self.is_open = False

def resource_generator():
    """
    使用yield创建上下文管理器的生成器
    """
    print("开始资源管理")
    yield ResourceManager("测试资源")
    print("资源管理结束")

def stream_simulator():
    """
    模拟流式数据的生成器
    """
    messages = ["Hello", "World", "Python", "Yield"]
    for message in messages:
        yield message
        # 模拟处理延迟
        import time
        time.sleep(0.1)

def batch_processor(items, batch_size=3):
    """
    批量处理数据的生成器
    """
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # 返回剩余的项
    if batch:
        yield batch 