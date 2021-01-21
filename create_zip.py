from zipfile import ZipFile


if __name__ == '__main__':
    with ZipFile("esrgan_scripts.zip", "w") as newzip:
        newzip.write("train.py")
        newzip.write("utils.py")
        newzip.write("models.py")
