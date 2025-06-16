 Phân loại hoa Iris bằng mô hình Cây Quyết Định (ID3)

Dự án này xây dựng một ứng dụng web đơn giản sử dụng **Streamlit** để phân loại loài hoa **Iris** dựa trên 4 đặc trưng hình thái học. Mô hình học máy được sử dụng là **Decision Tree Classifier** với tiêu chí **entropy (ID3)**.

---

 Mô tả dữ liệu

- Dataset: `iris.csv` gồm **150 mẫu**
- Mỗi mẫu gồm 4 đặc trưng:
  - Chiều dài và chiều rộng **đài hoa (sepal)**
  - Chiều dài và chiều rộng **cánh hoa (petal)**
- Nhãn: Tên loài hoa (`setosa`, `versicolor`, `virginica`)

---

 Mô hình và quy trình

- Mô hình: `DecisionTreeClassifier(criterion='entropy')`
- Tiền xử lý:
  - Chia dữ liệu: **64% train**, còn lại là validation và test
  - Chuẩn hóa đặc trưng bằng **StandardScaler**
- Huấn luyện và đánh giá mô hình

---

 Giao diện người dùng

Ứng dụng được xây dựng bằng **Streamlit**, với các tính năng:

- Nhập giá trị đặc trưng của hoa
- Hiển thị kết quả phân loại loài hoa
- Hiển thị hình ảnh minh họa
- Trực quan hóa dữ liệu bằng biểu đồ

---

 Thư viện sử dụng

- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `streamlit`

---

 Mục tiêu dự án

- Hỗ trợ học sinh/sinh viên hiểu cách hoạt động của **cây quyết định (decision tree)**
- Trực quan hóa quy trình phân loại dữ liệu một cách dễ hiểu

---

 Hướng dẫn chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py

