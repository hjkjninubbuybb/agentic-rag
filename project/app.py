from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    # 系统在这里默默加载模型，此时终端没有任何输出，你会觉得"卡住了"
    demo = create_gradio_ui()

    # 🟢 等上面全部加载完了，才会打印这一行
    print("\n🚀 Launching RAG Assistant...")
    demo.launch(css=custom_css)