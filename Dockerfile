FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# # 将依赖项安装到容器中
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# # 复制 FastAPI 应用的源代码到容器中
COPY main.py ./
# # 指定暴露的端口号
# EXPOSE 38006

# 运行应用
CMD ["python", "main.py"]
# CMD ["bash"]