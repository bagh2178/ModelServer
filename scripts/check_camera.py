import pyorbbecsdk  # 假设使用的是Orbbec官方的Python SDK


def get_camera_serial_number():
    # 初始化上下文以访问设备
    context = pyorbbecsdk.Context()

    # 获取设备列表
    device_list = context.query_devices()
    if not device_list:
        print("未检测到相机设备！")
        return

    # 检查设备列表是否为空
    if device_list.get_count() == 0:
        print("设备列表为空，无法获取设备信息！")
        return

    number_list = []
    for i in range(device_list.get_count()):
        # 获取第一个设备
        device = device_list.get_device_by_index(i)  # 使用get_device_by_index方法获取设备

        # 打开设备
        # device.open()

        # 获取设备信息
        device_info = device.get_device_info()

        # 获取序列号
        serial_number = device_info.get_serial_number()
        number_list.append(serial_number)

    # 关闭设备
    # device.close()

    return number_list


if __name__ == "__main__":
    serial_number = get_camera_serial_number()
    if serial_number:
        # print(f"相机序列号: {serial_number}")
        print(serial_number)