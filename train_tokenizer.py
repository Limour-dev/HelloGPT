from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers import Regex, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import emoji

tokenizer = Tokenizer(BPE(unk_token='<unk>', fuse_unk=True, byte_fallback=True))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),
    normalizers.Replace('\u0000', ' '),
    normalizers.Replace('\u000b', ' '),
    normalizers.Replace(Regex(r'\s+'), 'Ġ'),
    normalizers.Replace(Regex(r'(.)\1{3,}'), '…')
])

regex_unicode = r'''([\x00-\x40\x5B-\x60\x7B-\x7F]|[^\x00-\xffĠ]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5])+'''
regex_number = r'''\d|[零一二三四五六七八九十百千万亿]|\.|\+|\-'''
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split('Ġ', behavior='merged_with_next'),
    pre_tokenizers.Split('§', behavior='isolated'),
    pre_tokenizers.Split(Regex(regex_number), behavior='isolated'),
    pre_tokenizers.Split(Regex(regex_unicode), behavior='isolated'),
    pre_tokenizers.Split(Regex('Ġ*(\w+|[^\w\sĠ]+)'), behavior='isolated')
])
tokenizer.decoder = decoders.Sequence([
    decoders.Replace('Ġ', ' '),
    decoders.ByteFallback()
])

alphabet = [f'<0x{hex(i)[2:].upper():>02}>' for i in range(256)]
com_emoji = ['ლ', 'ღ', '‼', '⁉', '₽', '⃣', '™', 'ℹ', '←', '↑', '→', '↓', '↔', '↕', '↖', '↗', '↘', '↙', '↩', '↪', '∞',
             '⌚', '⌛', '⌨', '⏏', '⏩', '⏪', '⏫', '⏬', '⏭', '⏮', '⏯', '⏰', '⏱', '⏲', '⏳', '⏸', '⏹', '⏺', '▪', '▫', '▲',
             '▶', '▼', '◀', '○', '●', '◻', '◼', '◽', '◾', '☀', '☁', '☂', '☃', '☄', '★', '☎', '☑', '☔', '☕', '☘', '☚',
             '☛', '☜', '☝', '☞', '☟', '☠', '☢', '☣', '☦', '☪', '☫', '☬', '☮', '☯', '☰', '☱', '☲', '☳', '☴', '☵', '☶',
             '☷', '☸', '☹', '☺', '☻', '☼', '☽', '☾', '☿', '♀', '♂', '♅', '♆', '♇', '♈', '♉', '♊', '♋', '♌', '♍', '♎',
             '♏', '♐', '♑', '♒', '♓', '♔', '♕', '♖', '♗', '♟', '♠', '♡', '♣', '♥', '♦', '♧', '♨', '♩', '♪', '♫', '♬',
             '♻', '♾', '♿', '⚐', '⚑', '⚒', '⚓', '⚔', '⚕', '⚖', '⚗', '⚘', '⚙', '⚛', '⚜', '⚠', '⚡', '⚢', '⚣', '⚤', '⚥',
             '⚦', '⚧', '⚪', '⚫', '⚰', '⚱', '⚽', '⚾', '⚿', '⛄', '⛅', '⛇', '⛈', '⛎', '⛏', '⛑', '⛓', '⛔', '⛟', '⛩', '⛪',
             '⛰', '⛱', '⛲', '⛳', '⛴', '⛵', '⛶', '⛷', '⛸', '⛹', '⛽', '✂', '✅', '✉', '✊', '✋', '✌', '✍', '✏', '✒', '✔',
             '✖', '✝', '✡', '✨', '✳', '✴', '❄', '❇', '❌', '❎', '❓', '❗', '❣', '❤', '❥', '❦', '❧', '➕', '➖', '➗', '➡',
             '➰', '➿', '⬅', '⬆', '⬇', '⬛', '⬜', '⭐', '⭕', '「', '」', '『', '』', '〰', '〽', '㊗', '㊙', '️', '🀀', '🀁', '🀂',
             '🀃', '🀄', '🀅', '🀆', '🀇', '🀈', '🀉', '🀊', '🀋', '🀌', '🀍', '🀎', '🀏', '🀐', '🀑', '🀒', '🀓', '🀔', '🀕', '🀖', '🀗',
             '🀘', '🀙', '🀚', '🀛', '🀜', '🀝', '🀞', '🀟', '🀠', '🀡', '🀢', '🀣', '🀥', '🀦', '🀨', '🀩', '🀪', '🃏', '🅰', '🅱', '🅾',
             '🅿', '🆎', '🆑', '🆒', '🆓', '🆔', '🆕', '🆖', '🆗', '🆘', '🆙', '🆚', '🇦', '🇧', '🇨', '🇩', '🇪', '🇫', '🇬', '🇭', '🇮',
             '🇯', '🇰', '🇱', '🇲', '🇳', '🇴', '🇵', '🇶', '🇷', '🇸', '🇹', '🇺', '🇻', '🇼', '🇽', '🇾', '🇿', '🈁', '🈂', '🈚', '🈯',
             '🈲', '🈳', '🈴', '🈵', '🈶', '🈷', '🈸', '🈹', '🈺', '🉐', '🉑', '🌀', '🌁', '🌂', '🌃', '🌄', '🌅', '🌆', '🌇', '🌈', '🌉',
             '🌊', '🌋', '🌌', '🌍', '🌎', '🌏', '🌐', '🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘', '🌙', '🌚', '🌛', '🌜', '🌝', '🌞',
             '🌟', '🌠', '🌡', '🌢', '🌣', '🌤', '🌥', '🌦', '🌧', '🌨', '🌩', '🌪', '🌫', '🌬', '🌭', '🌮', '🌯', '🌰', '🌱', '🌲', '🌳',
             '🌴', '🌵', '🌶', '🌷', '🌸', '🌹', '🌺', '🌻', '🌼', '🌽', '🌾', '🌿', '🍀', '🍁', '🍂', '🍃', '🍄', '🍅', '🍆', '🍇', '🍈',
             '🍉', '🍊', '🍋', '🍌', '🍍', '🍎', '🍏', '🍐', '🍑', '🍒', '🍓', '🍔', '🍕', '🍖', '🍗', '🍘', '🍙', '🍚', '🍛', '🍜', '🍝',
             '🍞', '🍟', '🍠', '🍡', '🍢', '🍣', '🍤', '🍥', '🍦', '🍧', '🍨', '🍩', '🍪', '🍫', '🍬', '🍭', '🍮', '🍯', '🍰', '🍱', '🍲',
             '🍳', '🍴', '🍵', '🍶', '🍷', '🍸', '🍹', '🍺', '🍻', '🍼', '🍽', '🍾', '🍿', '🎀', '🎁', '🎂', '🎃', '🎄', '🎅', '🎆', '🎇',
             '🎈', '🎉', '🎊', '🎋', '🎌', '🎍', '🎎', '🎏', '🎐', '🎑', '🎒', '🎓', '🎕', '🎖', '🎗', '🎙', '🎚', '🎛', '🎞', '🎟', '🎠',
             '🎡', '🎢', '🎣', '🎤', '🎥', '🎦', '🎧', '🎨', '🎩', '🎪', '🎫', '🎬', '🎭', '🎮', '🎯', '🎰', '🎱', '🎲', '🎳', '🎴', '🎵',
             '🎶', '🎷', '🎸', '🎹', '🎺', '🎻', '🎼', '🎽', '🎾', '🎿', '🏀', '🏁', '🏂', '🏃', '🏄', '🏅', '🏆', '🏇', '🏈', '🏉', '🏊',
             '🏋', '🏌', '🏍', '🏎', '🏏', '🏐', '🏑', '🏒', '🏓', '🏔', '🏕', '🏖', '🏗', '🏘', '🏙', '🏚', '🏛', '🏜', '🏝', '🏞', '🏟',
             '🏠', '🏡', '🏢', '🏣', '🏤', '🏥', '🏦', '🏧', '🏨', '🏩', '🏪', '🏫', '🏬', '🏭', '🏮', '🏯', '🏰', '🏳', '🏴', '🏵', '🏸',
             '🏹', '🏺', '🏻', '🐀', '🐁', '🐂', '🐃', '🐄', '🐅', '🐆', '🐇', '🐈', '🐉', '🐊', '🐋', '🐌', '🐍', '🐎', '🐏', '🐐', '🐑',
             '🐒', '🐓', '🐔', '🐕', '🐖', '🐗', '🐘', '🐙', '🐚', '🐛', '🐜', '🐝', '🐞', '🐟', '🐠', '🐡', '🐢', '🐣', '🐤', '🐥', '🐦',
             '🐧', '🐨', '🐩', '🐪', '🐫', '🐬', '🐭', '🐮', '🐯', '🐰', '🐱', '🐲', '🐳', '🐴', '🐵', '🐶', '🐷', '🐸', '🐹', '🐺', '🐻',
             '🐼', '🐽', '🐾', '🐿', '👀', '👁', '👂', '👃', '👄', '👅', '👆', '👇', '👈', '👉', '👊', '👋', '👌', '👍', '👎', '👏', '👐',
             '👑', '👒', '👓', '👔', '👕', '👖', '👗', '👘', '👙', '👚', '👛', '👜', '👝', '👞', '👟', '👠', '👡', '👢', '👣', '👤', '👥',
             '👦', '👧', '👨', '👩', '👪', '👫', '👬', '👭', '👮', '👰', '👱', '👲', '👳', '👴', '👵', '👶', '👷', '👸', '👹', '👺', '👻',
             '👼', '👽', '👾', '👿', '💀', '💁', '💂', '💄', '💅', '💆', '💇', '💈', '💉', '💊', '💋', '💌', '💍', '💎', '💏', '💐', '💑',
             '💒', '💓', '💔', '💕', '💖', '💗', '💘', '💙', '💚', '💛', '💜', '💝', '💞', '💟', '💠', '💡', '💢', '💣', '💤', '💥', '💦',
             '💧', '💨', '💩', '💪', '💫', '💬', '💭', '💮', '💯', '💰', '💱', '💲', '💳', '💴', '💵', '💶', '💸', '💹', '💺', '💻', '💼',
             '💽', '💾', '💿', '📀', '📁', '📂', '📄', '📅', '📆', '📇', '📈', '📉', '📊', '📋', '📌', '📍', '📎', '📏', '📐', '📑', '📒',
             '📓', '📔', '📕', '📖', '📗', '📘', '📙', '📚', '📛', '📜', '📝', '📞', '📟', '📠', '📡', '📢', '📣', '📤', '📥', '📦', '📧',
             '📨', '📩', '📮', '📯', '📰', '📱', '📲', '📳', '📴', '📵', '📶', '📷', '📹', '📺', '📻', '📼', '📿', '🔀', '🔁', '🔂', '🔃',
             '🔄', '🔅', '🔆', '🔇', '🔈', '🔉', '🔊', '🔋', '🔌', '🔍', '🔎', '🔏', '🔐', '🔑', '🔒', '🔓', '🔔', '🔕', '🔖', '🔗', '🔘',
             '🔙', '🔚', '🔛', '🔜', '🔝', '🔞', '🔟', '🔠', '🔡', '🔢', '🔣', '🔤', '🔥', '🔦', '🔧', '🔨', '🔩', '🔪', '🔫', '🔬', '🔭',
             '🔮', '🔯', '🔰', '🔱', '🔲', '🔳', '🔴', '🔵', '🔶', '🔷', '🔸', '🔹', '🔺', '🔻', '🔼', '🔽', '🕆', '🕉', '🕊', '🕋', '🕌',
             '🕍', '🕎', '🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛', '🕜', '🕝', '🕞', '🕟', '🕠', '🕡', '🕢',
             '🕣', '🕤', '🕥', '🕦', '🕧', '🕭', '🕯', '🕰', '🕱', '🕳', '🕶', '🕷', '🕸', '🕹', '🕿', '🖂', '🖊', '🖋', '🖌', '🖍', '🖐',
             '🖒', '🖕', '🖖', '🖤', '🖥', '🖨', '🖱', '🖲', '🖼', '🗃', '🗄', '🗑', '🗒', '🗓', '🗔', '🗜', '🗝', '🗞', '🗡', '🗣', '🗦',
             '🗧', '🗨', '🗪', '🗯', '🗲', '🗳', '🗴', '🗶', '🗸', '🗹', '🗺', '🗻', '🗼', '🗽', '🗾', '🗿', '😀', '😁', '😂', '😃', '😄',
             '😅', '😆', '😇', '😈', '😉', '😊', '😋', '😌', '😍', '😎', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖', '😗', '😘', '😙',
             '😚', '😛', '😜', '😝', '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😦', '😧', '😨', '😩', '😪', '😫', '😬', '😭', '😮',
             '😯', '😰', '😱', '😲', '😳', '😴', '😵', '😶', '😷', '😸', '😹', '😺', '😻', '😼', '😽', '😾', '😿', '🙀', '🙁', '🙂', '🙃',
             '🙄', '🙅', '🙆', '🙇', '🙈', '🙉', '🙊', '🙋', '🙌', '🙍', '🙎', '🙏', '🚀', '🚁', '🚂', '🚃', '🚄', '🚅', '🚆', '🚇', '🚈',
             '🚉', '🚊', '🚋', '🚌', '🚍', '🚎', '🚏', '🚐', '🚑', '🚒', '🚓', '🚔', '🚕', '🚖', '🚗', '🚘', '🚙', '🚚', '🚛', '🚜', '🚝',
             '🚞', '🚟', '🚠', '🚡', '🚢', '🚣', '🚤', '🚥', '🚦', '🚧', '🚨', '🚩', '🚪', '🚫', '🚬', '🚭', '🚮', '🚯', '🚰', '🚱', '🚲',
             '🚳', '🚴', '🚵', '🚶', '🚸', '🚹', '🚺', '🚻', '🚼', '🚽', '🚾', '🚿', '🛀', '🛁', '🛂', '🛃', '🛄', '🛅', '🛆', '🛈', '🛉',
             '🛊', '🛋', '🛌', '🛍', '🛎', '🛏', '🛐', '🛑', '🛒', '🛕', '🛡', '🛢', '🛣', '🛤', '🛥', '🛧', '🛨', '🛩', '🛫', '🛬', '🛰',
             '🛲', '🛳', '🛴', '🛵', '🛶', '🛷', '🛸', '🛹', '🛺', '🟠', '🟡', '🟢', '🟣', '🟤', '🟥', '🟦', '🟧', '🟨', '🟩', '🟪', '🟫',
             '🤍', '🤎', '🤏', '🤐', '🤒', '🤓', '🤔', '🤕', '🤖', '🤗', '🤘', '🤙', '🤚', '🤛', '🤜', '🤝', '🤞', '🤟', '🤠', '🤡', '🤢',
             '🤣', '🤤', '🤥', '🤦', '🤧', '🤨', '🤩', '🤪', '🤫', '🤭', '🤮', '🤯', '🤰', '🤱', '🤲', '🤳', '🤴', '🤶', '🤷', '🤸', '🤹',
             '🤺', '🤼', '🤽', '🤾', '🤿', '🥀', '🥁', '🥂', '🥃', '🥄', '🥅', '🥇', '🥈', '🥉', '🥊', '🥋', '🥌', '🥍', '🥎', '🥏', '🥐',
             '🥑', '🥒', '🥓', '🥔', '🥕', '🥖', '🥗', '🥘', '🥙', '🥚', '🥛', '🥜', '🥝', '🥞', '🥟', '🥠', '🥡', '🥢', '🥣', '🥤', '🥥',
             '🥦', '🥧', '🥨', '🥩', '🥪', '🥫', '🥬', '🥭', '🥮', '🥯', '🥰', '🥱', '🥳', '🥴', '🥵', '🥶', '🥺', '🥻', '🥼', '🥽', '🥾',
             '🥿', '🦀', '🦁', '🦂', '🦃', '🦄', '🦅', '🦆', '🦇', '🦈', '🦉', '🦊', '🦋', '🦌', '🦍', '🦎', '🦏', '🦐', '🦑', '🦒', '🦓',
             '🦔', '🦕', '🦖', '🦗', '🦘', '🦙', '🦚', '🦛', '🦜', '🦝', '🦞', '🦟', '🦡', '🦢', '🦥', '🦦', '🦧', '🦨', '🦩', '🦪', '🦮',
             '🦯', '🦰', '🦱', '🦲', '🦳', '🦴', '🦵', '🦶', '🦷', '🦸', '🦺', '🦻', '🦼', '🦽', '🦾', '🦿', '🧀', '🧁', '🧂', '🧃', '🧄',
             '🧅', '🧆', '🧇', '🧈', '🧉', '🧊', '🧐', '🧒', '🧓', '🧖', '🧞', '🧟', '🧠', '🧡', '🧢', '🧣', '🧤', '🧥', '🧦', '🧧', '🧨',
             '🧩', '🧪', '🧫', '🧬', '🧭', '🧮', '🧯', '🧰', '🧱', '🧲', '🧳', '🧴', '🧵', '🧶', '🧷', '🧸', '🧹', '🧺', '🧻', '🧼', '🧽',
             '🧾', '🧿', '🩰', '🩱', '🩲', '🩳', '🩸', '🩹', '🩺', '🪀', '🪁', '🪂', '🪐', '🪑', '🪒', '🪓', '🪔', '🪕', '🪡']
trainer = BpeTrainer(special_tokens=['<unk>', '§', '<eos>'],
                     vocab_size=32765 - 256,
                     min_frequency=3,
                     initial_alphabet=com_emoji,
                     max_token_length=5,
                     show_progress=True
                     )

if False:
    import os

    filePath = r'D:\datasets\h-corpus'
    filePath = [os.path.join(filePath, f) for f in os.listdir(filePath) if f.endswith('txt')][:1000]
else:
    filePath = [r'tmp_jieba.final.txt']

tokenizer.train(filePath, trainer=trainer)

tokenizer.save("HelloBPE.tokenizer.json", pretty=True)

import json

with open('HelloBPE.tokenizer.json', 'r', encoding='utf-8') as f:
    x = json.load(f)
y = {'<unk>': 0, '§': 1, '<eos>': 2}
for i, v in enumerate(alphabet):
    y[v] = i + 3
for k, v in x['model']['vocab'].items():
    if k not in ['<unk>', '§', '<eos>']:
        y[k] = v + 256
x['model']['vocab'] = y
with open('HelloBPE.tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(x, f, indent=2, ensure_ascii=False)

tokenizer = Tokenizer.from_file("HelloBPE.tokenizer.json")
tmp = tokenizer.encode('不了，我更喜欢那种面对神秘的期待感，连同你对于我们旅途的安排。')
print(tmp.tokens, '\n', tokenizer.decode(tmp.ids))
print(tokenizer.normalizer.normalize_str('hello   world'))
print(tokenizer.pre_tokenizer.pre_tokenize_str('GaaaĠaaa你好'))
print(tokenizer.encode(
    '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文 ，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词').tokens)
print(tokenizer.encode(
    '在被问到锡兰还有博士更喜欢谁的时候，黑支支吾吾地不愿意说话。原因也很简单，此时的她刚刚结束与自己侍奉的大小姐和博士双飞之后的迷情，被一起同登极乐的两个人一起这么问，自然不知道应该怎么回答。结果作为惩罚，她被还没有玩够的博士按着猛烈进入，被锡兰亲吻着磨蹭着下体，然后被两人一起狠狠地做到不知高潮了几次，直到意识模糊昏睡到了次日。').tokens)
# with open(r'tmp_jieba.final.txt', 'r', encoding='utf-8') as f:
#     tmp2 = tokenizer.pre_tokenizer.pre_tokenize_str(f.read()[-10000:])
