# Only for creation of files
# KEEP ATTENTION: that writer is called with an append

if __name__ == '__main__' :
    from OwnUtils.xL_CustomExtractor import CustomExtractor
    rec = CustomExtractor()
    rec.create_validation_test_files(True, True)
    print("END")