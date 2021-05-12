from model import User, Simsimi, KoGPT2Chat 

def chatbot(input : User)->Simsimi : 
    """심심할 때 심심이와 대화해보세요."""
    model = KoGPT2Chat.load_from_checkpoint('model_chp/model_-last.ckpt')
    text = input.user
    return Simsimi(simsimi = model.chat(text = text.strip()))

 