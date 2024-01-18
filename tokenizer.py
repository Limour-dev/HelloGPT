from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("HelloBPE.tokenizer.json")

if __name__ == '__main__':
    print(tokenizer.encode(
        '你好，请问你是？').tokens)
    print(tokenizer.encode(
        '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文 ，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词').tokens)
    print(tokenizer.encode(
        '在被问到锡兰还有博士更喜欢谁的时候，黑支支吾吾地不愿意说话。原因也很简单，此时的她刚刚结束与自己侍奉的大小姐和博士双飞之后的迷情，被一起同登极乐的两个人一起这么问，自然不知道应该怎么回答。结果作为惩罚，她被还没有玩够的博士按着猛烈进入，被锡兰亲吻着磨蹭着下体，然后被两人一起狠狠地做到不知高潮了几次，直到意识模糊昏睡到了次日。').tokens)