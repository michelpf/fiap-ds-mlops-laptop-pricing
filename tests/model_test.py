from pathlib import Path
import joblib

def prepare_data(data):

    data_processed = []

    data_processed.append(1) if data["brand"] == "asus" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "dell" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "hp" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "lenovo" else data_processed.append(0)
    data_processed.append(1) if data["brand"] == "other" else data_processed.append(0)

    data_processed.append(1) if data["processor_brand"] == "amd" else data_processed.append(0)
    data_processed.append(1) if data["processor_brand"] == "intel" else data_processed.append(0)
    data_processed.append(1) if data["processor_brand"] == "m1" else data_processed.append(0)
    
    data_processed.append(1) if data["processor_name"] == "core i3" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "core i5" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "core i7" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "other" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "ryzen 5" else data_processed.append(0)
    data_processed.append(1) if data["processor_name"] == "ryzen 7" else data_processed.append(0)
    
    data_processed.append(1) if data["os"] == "other" else data_processed.append(0)
    data_processed.append(1) if data["os"] == "windows" else data_processed.append(0)
    
    data_processed.append(1) if data["weight"] == "casual" else data_processed.append(0)
    data_processed.append(1) if data["weight"] == "gaming" else data_processed.append(0)
    data_processed.append(1) if data["weight"] == "thinnlight" else data_processed.append(0)

    data_processed.append(1) if data["warranty"] == "0" else data_processed.append(0)
    data_processed.append(1) if data["warranty"] == "1" else data_processed.append(0)
    data_processed.append(1) if data["warranty"] == "2" else data_processed.append(0)
    data_processed.append(1) if data["warranty"] == "3" else data_processed.append(0)

    data_processed.append(1) if data["touchscreen"] == "0" else data_processed.append(0)
    data_processed.append(1) if data["touchscreen"] == "1" else data_processed.append(0)
    
    data_processed.append(1) if data["ram_gb"] == "4" else data_processed.append(0)
    data_processed.append(1) if data["ram_gb"] == "8" else data_processed.append(0)
    data_processed.append(1) if data["ram_gb"] == "16" else data_processed.append(0)
    data_processed.append(1) if data["ram_gb"] == "32" else data_processed.append(0)

    data_processed.append(1) if data["hdd"] == "0" else data_processed.append(0)
    data_processed.append(1) if data["hdd"] == "512" else data_processed.append(0)
    data_processed.append(1) if data["hdd"] == "1024" else data_processed.append(0)
    data_processed.append(1) if data["hdd"] == "2048" else data_processed.append(0)

    data_processed.append(1) if data["ssd"] == "0" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "128" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "256" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "512" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "1024" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "2048" else data_processed.append(0)
    data_processed.append(1) if data["ssd"] == "3072" else data_processed.append(0)

    data_processed.append(1) if data["graphic_card"] == "4" else data_processed.append(0)
    data_processed.append(1) if data["graphic_card"] == "8" else data_processed.append(0)
    data_processed.append(1) if data["graphic_card"] == "16" else data_processed.append(0)
    data_processed.append(1) if data["graphic_card"] == "32" else data_processed.append(0)

    data_processed.append(1) if data["ram_type"] == "ddr4" else data_processed.append(0)
    data_processed.append(1) if data["ram_type"] == "other" else data_processed.append(0)

    data_processed.append(1) if data["os_bit"] == "32" else data_processed.append(0)
    data_processed.append(1) if data["os_bit"] == "64" else data_processed.append(0)

    return data_processed

def range_golden_data(data, target):
    return target < data < target*1.3

def test_model_exists():
    arquivo_path = Path("models/model.pkl")
    assert arquivo_path.is_file(), "model exists in the expected path"

def test_golden_data():
    model = joblib.load("models/model.pkl")
    
    payload = {
        "brand": "dell",
        "processor_brand": "intel",
        "processor_name": "core i5",
        "os": "windows",
        "weight": "casual",
        "warranty": "2",
        "touchscreen": "0",
        "ram_gb": "16",
        "hdd": "0",
        "ssd": "256",
        "graphic_card": "8",
        "ram_type": "ddr4",
        "os_bit": "64"
    }

    data_processed = prepare_data(payload)

    result = model.predict([data_processed])

    result = int(result[0])

    print(result)

    assert range_golden_data(result, 66000), "ensuring golden data results"


def test_model_load_call():
    model = joblib.load("models/model.pkl")
    
    payload = {
        "brand": "dell",
        "processor_brand": "intel",
        "processor_name": "core i5",
        "os": "windows",
        "weight": "casual",
        "warranty": "2",
        "touchscreen": "0",
        "ram_gb": "16",
        "hdd": "0",
        "ssd": "256",
        "graphic_card": "8",
        "ram_type": "ddr4",
        "os_bit": "64"
    }

    data_processed = prepare_data(payload)

    result = model.predict([data_processed])

    result = int(result[0])

    assert isinstance(result, int), "ensuring data type"
    assert result >= 0, "ensuring data value coehrence"
