import PacketPreprocess
import DetectModels

if __name__ == "__main__":
    ppc = PacketPreprocess()
    ppc.loadCsv()
    ppc.missingValue()
    lstmZip_df = ppc.normalizeToLstm()