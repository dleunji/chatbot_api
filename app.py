from model import User, Comcom, KoGPT2Chat 

def chatbot(input : User)->Comcom : 
    """심심할 때 컴컴이와 대화해보세요."""
    model = KoGPT2Chat.load_from_checkpoint('model_chp/model_-last.ckpt')
    text = input.user
    return Comcom(comcom = model.chat(text = text.strip()))

 