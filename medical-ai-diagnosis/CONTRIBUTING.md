# Hướng dẫn đóng góp (CONTRIBUTING)

## 1. Nhánh (Branching)

| Nhánh | Mục đích |
|---|---|
| `main` | Code ổn định, đã review, sẵn sàng demo/nộp bài |
| `dev` | Nhánh tích hợp, các feature merge vào đây trước |
| `feature/<ten>` | Phát triển tính năng mới (VD: `feature/xray-classifier`) |
| `fix/<ten>` | Sửa lỗi (VD: `fix/dataloader-crash`) |
| `experiment/<ten>` | Thử nghiệm model/hyperparameter, không ảnh hưởng code chính |

Quy trình:
1. Tạo nhánh mới từ `dev`: `git checkout -b feature/ten-tinh-nang dev`
2. Code + commit theo quy ước bên dưới.
3. Push nhánh, mở Pull Request vào `dev`.
4. Ít nhất 1 thành viên khác review trước khi merge.
5. Định kỳ merge `dev` vào `main` khi đã ổn định.

## 2. Quy ước commit message

Dùng dạng: `<loại>: <mô tả ngắn>`

- `feat:` thêm tính năng mới
- `fix:` sửa lỗi
- `docs:` thay đổi tài liệu
- `refactor:` tái cấu trúc code, không đổi hành vi
- `experiment:` thử nghiệm model/tham số
- `chore:` việc lặt vặt (cập nhật dependency, config...)

Ví dụ: `feat: add ResNet50 classifier for chest X-ray`

## 3. Pull Request

- Title rõ ràng, mô tả thay đổi + cách test.
- Liên kết issue liên quan (nếu có): `Closes #12`.
- Không merge PR của chính mình khi chưa có review (trừ trường hợp gấp và đã trao đổi trước trong nhóm).

## 4. Quy ước code

- Đặt tên biến/hàm bằng tiếng Anh, rõ nghĩa.
- Mỗi module trong `src/` nên có docstring ngắn mô tả chức năng.
- Ưu tiên tái sử dụng hàm/class đã có trong `src/utils`, `src/data`... trước khi tạo mới.
- Không hardcode đường dẫn dữ liệu, tham số model — đưa vào `configs/*.yaml`.
- Không commit file dữ liệu, checkpoint model (`.pt`, `.h5`, `.ckpt`), API key, `.env`.

## 5. Phân công vai trò gợi ý

- **Data**: thu thập, tiền xử lý, augmentation, quản lý dataset.
- **Model**: xây dựng và huấn luyện model, tuning.
- **Backend/API**: đóng gói model thành API, tích hợp hệ thống.
- **Evaluation & Docs**: đánh giá kết quả, viết báo cáo/tài liệu.

Một thành viên có thể đảm nhiệm nhiều vai trò tùy quy mô nhóm.
