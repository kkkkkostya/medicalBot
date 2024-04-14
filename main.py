import numpy
import telebot
import torch
from PIL import Image
from telebot import types
from torch import nn
from torchvision.transforms import v2
from pneumaniaModule.pneumaniaFuntions import pneumoniaNet
from secondNetModule.chestXrayNetFunctions import chestXrayNet
from io import BytesIO
import torchvision.transforms.v2 as T
from interpretation_methods.interpretation_methods import intepretationLRP
from statisticalModule.stat_methods import empirical_p_values
import numpy as np
import captum
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

diseases = np.array(
    ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
     'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'])
bot = telebot.TeleBot('7048035782:AAFHedcWRB9fHSiXWL9xYHaTxe1-ZVjq5D8')
first_disease = 'Пневмония'
second_disease = 'Диагностика других заболеваний'

pneumaniaModel = pneumoniaNet(pretrained=True)
chestXrayModel = chestXrayNet(pretrained=True)

chat_states = {}


@bot.message_handler(
    content_types=["text", "audio", "document", "photo", "sticker", "video", "video_note", "voice", "location",
                   "contact",
                   "new_chat_members", "left_chat_member", "new_chat_title", "new_chat_photo", "delete_chat_photo",
                   "group_chat_created", "supergroup_chat_created", "channel_chat_created", "migrate_to_chat_id",
                   "migrate_from_chat_id", "pinned_message"])
def check_first_message(message):
    chat_id = message.chat.id

    if chat_id in chat_states:
        # Если состояние "ожидание ответа", то игнорируем сообщение
        if chat_states[chat_id] == "wait_for_answer":
            return

    chat_states[chat_id] = "wait_for_answer"

    if message.content_type == 'text' and message.text == "/start":
        start_function(message)
    else:
        bot.send_message(chat_id, "Чтобы начать работу с ботом, выберите в меню start")
        bot.register_next_step_handler(message, check_first_message)
        chat_states[chat_id] = None


@bot.message_handler(commands=['start'])
def start_function(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_First_net = types.KeyboardButton(first_disease)
    button_Second_net = types.KeyboardButton(second_disease)
    markup.row(button_First_net, button_Second_net)
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}')
    bot.send_message(message.chat.id, f'С помощью кнопок в меню выберите желаемую диагностику', reply_markup=markup)
    bot.register_next_step_handler(message, on_click_start)


def on_click_start(message):
    if message.content_type != 'text' or message.text not in [first_disease, second_disease]:
        bot.send_message(message.chat.id, "Такой команды к сожалению нет(")
        bot.register_next_step_handler(message, on_click_start)
    else:
        example_photo = open('images/IM-0017-0001.jpeg', 'rb') if message.text == first_disease else open(
            'images/00003928_000.png', 'rb')
        bot.send_message(message.chat.id, 'Отлично, теперь нужно отправить рентген легких для определения {des}'.format(
            des='пневмонии' if message.text == first_disease else 'заболеваний'))
        bot.send_message(message.chat.id, 'Изображение должно выглядить вот так:')
        bot.send_photo(message.chat.id, example_photo)
        bot.register_next_step_handler(message, lambda m: click_disease_button(m, message.text))


def click_disease_button(message, disease_name):
    wrong_message = 'Неправильный формат ввода, нужно прислать изображение с рентгеном легких'
    success = "Отлично, отправьте рентген легких"
    func = pheumaniaDisease if disease_name == first_disease else multiDisease
    second_net = True if disease_name == second_disease else False
    if message.content_type == 'text':
        if message.text in [first_disease, second_disease]:
            bot.send_message(message.chat.id, success)
            on_click_start(message)
        elif message.text == '/start':
            start_function(message)
        else:
            bot.send_message(message.chat.id, wrong_message)
            bot.register_next_step_handler(message, lambda m: click_disease_button(m, disease_name))
    elif message.content_type != 'photo':
        if message.content_type == 'document' and message.document.mime_type.startswith('image'):
            func(message, get_image(message, True, second_net))
        else:
            bot.send_message(message.chat.id, wrong_message)
            bot.register_next_step_handler(message, lambda m: click_disease_button(m, disease_name))
    else:
        func(message, get_image(message, False, second_net))


@bot.message_handler(
    content_types=['photo'])
def pheumaniaDisease(message, image):
    with torch.no_grad():
        pneumaniaModel.eval()
        logit = pneumaniaModel(image[None, :])[0]
    pred = numpy.argmax(logit.detach().numpy()).item()
    bot.send_message(message.chat.id, 'Результаты классификации:')
    bot.send_message(message.chat.id,
                     'Вероятнее всего на фото {p} пневмония'.format(p='присутствует' if pred else 'отсутствует'))
    if pred == 0:
        bot.send_message(message.chat.id,
                         'Оценка вероятности ошибки предсказания: {p} '.format(
                             p=empirical_p_values(np.array([logit[1].detach().numpy()]))[0]))

    interpritate(message, 0, image, pred)
    bot.register_next_step_handler(message, lambda m: click_disease_button(m, first_disease))


@bot.message_handler(
    content_types=['photo'])
def multiDisease(message, image):
    with torch.no_grad():
        chestXrayModel.eval()
        pred = (nn.Sigmoid()(chestXrayModel(image[None, :])[0])) > 0.5
    bot.send_message(message.chat.id, 'Результаты классификации:')
    if len(diseases[pred]) == 0:
        bot.send_message(message.chat.id, 'Модель не выявила никаких заболеваний')
    else:
        bot.send_message(message.chat.id,
                         'Вероятнее всего на фото присутствуют следующие заболевания: {disease}'.format(
                             disease=', '.join(diseases[pred])))
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    interpret_buttons = []
    for i in range(len(diseases)):
        interpret_buttons.append(types.KeyboardButton(diseases[i]))
        if i != 0 and (i + 1) % 4 == 0:
            markup.row(interpret_buttons[-4], interpret_buttons[-3], interpret_buttons[-2], interpret_buttons[-1])
    markup.row(interpret_buttons[-2], interpret_buttons[-1])
    bot.send_message(message.chat.id, 'Выберите интерпретацию одной из болезней, нажав соответствующую кнопку',
                     reply_markup=markup)
    bot.register_next_step_handler(message, lambda m: button_hander(m, image))


def get_image(message, is_doc=False, second_net=False):
    if is_doc:
        file_info = bot.get_file(message.document.file_id)
    else:
        file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    return get_tensor_from_bytes(downloaded_file, second_net)


def button_hander(message, image):
    if message.content_type == 'photo':
        multiDisease(message, get_image(message, False, True))
    elif message.content_type != 'text' or (message.text not in diseases and message.text != '/start'):
        bot.send_message(message.chat.id, 'Неправильный ввод(')
        bot.register_next_step_handler(message, lambda m: button_hander(m, image))
    elif message.text == '/start':
        start_function(message)
    else:
        interpritate(message, 1, image, int(np.where(diseases == message.text)[0]))
        bot.register_next_step_handler(message, lambda m: button_hander(m, image))


def interpritate(message, model_type, image, target):
    bot.send_message(message.chat.id, 'Интерпретация модели')
    interp = intepretationLRP(chestXrayModel if model_type else pneumaniaModel, image[None, :], target)
    # interp = intepretationGradCam(chestXrayModel if model_type else pneumaniaModel, model_type, image[None, :], target)  GradCam interpritation
    origin_image = np.transpose(image.detach().numpy().squeeze(), (1, 2, 0))
    attr_img, _ = captum.attr.visualization.visualize_image_attr(interp, method='blended_heat_map', cmap='Reds',
                                                                 original_image=origin_image)
    img_byte_arr = BytesIO()
    attr_img.savefig(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    plt.close(attr_img)

    # Отправка изображения пользователю
    bot.send_photo(message.chat.id, photo=img_byte_arr)


def get_tensor_from_bytes(b, second_net=False):
    """
    Gets the PIL.Image object from the bytes of photo
    :param second_net:
    :param b: Bytes of the photo
    :type b: bytes
    """
    if second_net:
        test_transform = T.Compose([
            T.Resize((1024, 1024)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        test_transform = T.Compose([
            T.Resize((255, 255)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            T.Grayscale(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    stream = BytesIO(b)
    image = Image.open(stream)
    image = test_transform(image)
    stream.close()
    return image


bot.infinity_polling()
