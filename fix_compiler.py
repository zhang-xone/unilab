import xml.etree.ElementTree as ET

path = "src/unilab/assets/robots/cod_2026_balance/cod_2026_balance.xml"
tree = ET.parse(path)
root = tree.getroot()

# 移除多余的 compiler 标签
compilers = root.findall("compiler")
if len(compilers) > 1:
    for comp in compilers[1:]:
        root.remove(comp)
    
    # 确保保留的 compiler 有正确的 meshdir
    compilers[0].set("meshdir", "assets")

ET.indent(tree, space="  ")
tree.write(path, encoding="unicode")
print("✅ 已修复重复的 compiler 标签")
