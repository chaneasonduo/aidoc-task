import pytest
from pytests.yield_example import (
    ResourceManager,
    batch_processor,
    fibonacci_generator,
    number_generator,
    resource_generator,
    stream_simulator,
)


class TestYieldExamples:
    """测试yield用法的各种示例"""
    
    def test_number_generator(self):
        """测试数字生成器"""
        gen = number_generator(5)
        print("gen type: ",type(gen))
        result = list(gen)
        assert result == [0, 1, 2, 3, 4]
    
    def test_fibonacci_generator(self):
        """测试斐波那契数列生成器"""
        gen = fibonacci_generator(8)
        result = list(gen)
        assert result == [0, 1, 1, 2, 3, 5, 8, 13]
    
    def test_generator_iteration(self):
        """测试生成器的迭代特性"""
        gen = number_generator(3)
        
        # 第一次迭代
        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 2
        
        # 生成器应该已经耗尽
        with pytest.raises(StopIteration):
            next(gen)
    
    def test_generator_memory_efficiency(self):
        """测试生成器的内存效率"""
        # 生成器不会一次性生成所有数据
        gen = number_generator(1000000)
        # 只获取前几个值，不会占用大量内存
        first_few = [next(gen) for _ in range(5)]
        assert first_few == [0, 1, 2, 3, 4]
    
    def test_resource_manager(self):
        """测试资源管理器"""
        with ResourceManager("测试文件") as rm:
            assert rm.is_open == True
            assert rm.resource_name == "测试文件"
        
        # 离开上下文后应该关闭
        assert rm.is_open == False
    
    def test_resource_generator(self):
        """测试资源生成器"""
        # 使用生成器作为上下文管理器
        for resource in resource_generator():
            assert isinstance(resource, ResourceManager)
            assert resource.resource_name == "测试资源"
            break  # 只测试一次
    
    def test_stream_simulator(self):
        """测试流式数据模拟器"""
        gen = stream_simulator()
        messages = list(gen)
        assert messages == ["Hello", "World", "Python", "Yield"]
    
    def test_batch_processor(self):
        """测试批量处理器"""
        items = [1, 2, 3, 4, 5, 6, 7]
        batches = list(batch_processor(items, batch_size=3))
        
        assert len(batches) == 3
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7]
    
    def test_batch_processor_empty(self):
        """测试空数据的批量处理"""
        batches = list(batch_processor([], batch_size=3))
        assert batches == []
    
    def test_batch_processor_small_batch(self):
        """测试小于批次大小的数据处理"""
        items = [1, 2]
        batches = list(batch_processor(items, batch_size=3))
        assert batches == [[1, 2]]

class TestYieldAdvancedFeatures:
    """测试yield的高级特性"""
    
    def test_generator_expression(self):
        """测试生成器表达式"""
        # 生成器表达式
        gen_expr = (x * x for x in range(5))
        result = list(gen_expr)
        assert result == [0, 1, 4, 9, 16]
    
    def test_yield_from(self):
        """测试yield from语法"""
        def sub_generator():
            yield 1
            yield 2
        
        def main_generator():
            yield from sub_generator()
            yield 3
        
        result = list(main_generator())
        assert result == [1, 2, 3]
    
    def test_generator_send(self):
        """测试生成器的send方法"""
        def counter():
            count = 0
            while True:
                received = yield count
                if received is not None:
                    count = received
                else:
                    count += 1
        
        gen = counter()
        next(gen)  # 启动生成器
        
        assert gen.send(None) == 1
        assert gen.send(10) == 10
        assert gen.send(None) == 11

if __name__ == "__main__":
    # 可以直接运行这个文件来执行测试
    pytest.main([__file__, "-v"]) 