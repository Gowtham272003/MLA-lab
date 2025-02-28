import asyncio
from googletrans import Translator, LANGUAGES


async def translate_text(text, target_languages):
    translator = Translator()
    translations = {}

    for lang in target_languages:
        translated = await translator.translate(text, dest=lang)
        translations[LANGUAGES[lang]] = translated.text

    return translations


async def main():
    text_to_translate = input("enter a sentence:")
    target_languages = [ 'te',   'es', 'fr', 'de', 'zh-cn','ja','ar', 'it', 'pt', 'ru', 'ko','hi',  'bn', 'id', 'mr', 'tr',]

 # Telugu, Spanish, French, German, Chinese, Japanese

    translations = await translate_text(text_to_translate, target_languages)

    for lang, translation in translations.items():
        print(f"{lang.capitalize()}: {translation}")


if __name__ == "__main__":
    asyncio.run(main())
