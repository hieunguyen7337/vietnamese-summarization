from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./vit5_summary_model")  
model = AutoModelForSeq2SeqLM.from_pretrained("./vit5_summary_model")

input_text = """vietnews: 
Quốc hội sáng nay thảo luận về dự thảo Luật Đấu thầu (sửa đổi), trong đó, nội dung được nhiều đại biểu quan tâm là quy định về trường hợp được chỉ định gói thầu.
Điều 23 của dự thảo luật này quy định, được chỉ định gói thầu dịch vụ tư vấn, thuốc, thiết bị y tế, vật tư y tế... trong trường hợp "cần triển khai ngay để phục vụ công tác phòng, chống dịch bệnh, tránh gây nguy hại đến tính mạng, sức khỏe người dân". Theo Giám đốc Sở Y tế Hà Nội Trần Thị Nhị Hà, quy định này là cần thiết nhưng chưa rõ thế nào là "gói thầu cần triển khai ngay", có thể dẫn đến nguy cơ áp dụng tùy tiện hình thức chỉ định thầu.
Bà Hà nói cụm từ "cần triển khai ngay" được quy định từ Luật Đấu thầu năm 2013 đã gây ra sự lúng túng trong bối cảnh dịch bệnh. Một số đơn vị áp dụng hình thức chỉ định thầu đã bị xác định vi phạm trong việc lựa chọn hình thức đấu thầu. Do đó, đại biểu đề nghị cụ thể hóa khái niệm này.
Tương tự, Điều 23 cũng đề cập "được chỉ định thầu với gói mua sắm thuốc, hóa chất, thiết bị y tế, vật tư y tế để cấp cứu người bệnh theo quy định của Luật khám bệnh, chữa bệnh (sửa đổi)".
Nhưng Đại biểu Trần Khánh Thu băn khoăn, Luật khám bệnh, chữa bệnh (sửa đổi) lại không có quy định nào về vấn đề này. Bà đề nghị quy định rõ hơn thế nào là "trường hợp cấp bách trong y tế, cơ quan nào có thẩm quyền xác định trường cấp bách".
Vi phạm chủ yếu trong mua sắm, đấu thầu thời gian qua đều bắt nguồn từ giá gói thầu. Tuy nhiên, việc xác định giá theo hướng dẫn tại Thông tư 68 của Bộ Tài chính đang nhiều bất cập.
Một trong các phương thức xác định giá gói thầu là sử dụng 3 báo giá. Tuy nhiên, phương pháp này, theo bà Trần Thị Nhị Hà, "không bảo đảm giá hàng hóa là giá thị trường". Bởi đây không phải là giá giao dịch thành công và cạnh tranh. Đại biểu đoàn Hà Nội đề nghị ban soạn thảo cân nhắc quy định nguyên tắc xác định giá gói thầu ngay trong dự thảo, tạo cơ sở pháp lý cho Chính phủ quy định cụ thể.
Trong khi đó, đại biểu Nguyễn Hữu Chính đồng tình với ban soạn thảo khi cho áp dụng quy định chỉ định với gói thầu mua thuốc, thiết bị y tế chỉ duy nhất có một hãng sản xuất trên thị trường do yêu cầu giải pháp công nghệ. Tuy nhiên, để tránh bị lạm dụng, đại biểu này kiến nghị quy định chặt chẽ chi tiết tiêu chí, điều kiện áp dụng, hình thức chỉ định thầu. Ông Chính cũng nhất trí với dự thảo về trường hợp được chỉ định thầu, trong đó bổ sung trường hợp gói thầu cung cấp thuốc, vật tư để cấp cứu người bệnh trong trường hợp cơ sở khám bệnh, chữa bệnh không đủ.
Đại biểu Phạm Thị Kiều nhìn nhận, qua đại dịch Covid-19 đã cho thấy năng lực đáp ứng và tiếp cận vật tư y tế, hóa chất, sinh phẩm, vaccine, trang thiết bị còn nhiều hạn chế do sản xuất trong nước chưa đáp ứng. Quy định về quản lý, đấu thầu trang thiết bị y tế còn nhiều bất cập.
Để tháo gỡ, bà Kiều đề nghị điều chỉnh dự thảo theo hướng "khi có tình huống khẩn cấp, tổ chức được giao mua sắm có thể ứng trước hàng hóa để phục vụ đúng mục đích, yêu cầu cấp bách theo chỉ đạo của các cấp có thẩm quyền, sau đó thực hiện quy trình chỉ định thầu rút gọn theo quy định".</s>"""

# input_text =  "vietnews: " + sentence + " </s>"
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

outputs = model.generate(input_ids=input_ids,
                         max_length=256,
                         early_stopping=True)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)