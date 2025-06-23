import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

from brom import БромКлиент

# Load environment variables
load_dotenv()

def send_file(folder):
    """Отправка бинарных сведений в УЗ"""

    UZ3 = БромКлиент(
        os.getenv('BROM_URL', ''),
        os.getenv('BROM_USERNAME', ''),
        os.getenv('BROM_PASSWORD', '')
    )
    BitsData = 0
    for filename in os.listdir(folder):
        unique_key = None
        print(1, filename)
        filepath = Path(folder, filename)
        if filepath.is_file() and filename.lower().endswith('.pdf'):
            unique_key = uuid.uuid4()
            with open(Path(folder, 'keys.txt'), 'a') as f:
                f.write(f'{Path(filename).stem}. {unique_key}\n')
            print(f'{unique_key} -----------------------------------------')
            with open(Path(folder, filename), 'rb') as openFile:
                BitsData = openFile.read()
            UZ3.РаботаСИнструментамиОбмена.ДобавитьБинарноеСведениеПоЗаданию(unique_key,
                                                                             'PDF_section', BitsData, filename)
            section_info_folder_path = Path(folder, Path(filename).stem)
            for filename_inside in os.listdir(section_info_folder_path):
                print(2, filename_inside)
                with open(Path(section_info_folder_path, filename_inside), 'rb') as openFile:
                    BitsData_inside = openFile.read()
                UZ3.РаботаСИнструментамиОбмена.ДобавитьБинарноеСведениеПоЗаданию(unique_key,
                                                                                 Path(filename_inside).stem, BitsData_inside,
                                                                                 filename_inside)
