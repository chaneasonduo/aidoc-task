import time

def fetch_data_sync(i):
    print(f"开始获取数据 {i}")
    time.sleep(3)  # 阻塞3秒
    print(f"获取完成 {i}")

def main_sync():
    for i in range(3):
        fetch_data_sync(i)

main_sync()

##
print("异步")

import asyncio

async def fetch_data_async(i):
    print(f"开始获取数据 {i}")
    await asyncio.sleep(3)  # 非阻塞地等待3秒
    print(f"获取完成 {i}")

async def main_async():
    tasks = [fetch_data_async(i) for i in range(3)]
    await asyncio.gather(*tasks)  # 并发执行3个任务

asyncio.run(main_async())
