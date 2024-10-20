import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 시뮬레이션 파라미터
frequency = 5  # Hz, 오실레이터의 기본 주파수
injection_strength = 0.5  # 주입된 파형의 강도
phase_shift = np.pi / 4  # 오실레이터와 주입된 파형 사이의 위상 차이
amplitude = 1  # 오실레이터의 진폭
duration = 2  # 초
sampling_rate = 1000  # 초당 샘플 수

# 유도된 파라미터
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# 오실레이터 파형
oscillator_wave = amplitude * np.sin(2 * np.pi * frequency * t)

# 주입된 (반사된) 파형 (단순히 위상 차이와 스케일링을 적용)
injected_wave = injection_strength * amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)

# 결합된 파형 (오실레이터 + 주입된 파형)
combined_wave = oscillator_wave + injected_wave

# 그림과 축 생성
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)  # 파형의 1초를 표시
ax.set_ylim(-2 * amplitude, 2 * amplitude)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title('Self-Injection Locking Simulation')

# 오실레이터, 주입된, 결합된 파형을 위한 초기 라인 설정
line_osc, = ax.plot([], [], label='Oscillator Wave', color='blue')
line_inj, = ax.plot([], [], label='Injected Wave', color='orange')
line_combined, = ax.plot([], [], label='Combined Wave', color='green')

# 현재 시간을 나타내는 수직선
current_time_line = ax.axvline(x=0, color='red', linestyle='--', label='Current Time')

# 범례 설정 (중복 제거)
ax.legend(loc='upper right')

# 프레임 수와 윈도우 크기 설정
frames = len(t)
window_size = int(sampling_rate * 0.02)  # 20 ms 윈도우

def init():
    """애니메이션의 초기 상태를 설정합니다."""
    line_osc.set_data([], [])
    line_inj.set_data([], [])
    line_combined.set_data([], [])
    current_time_line.set_xdata([0, 0])  # 시퀀스로 변경
    return [line_osc, line_inj, line_combined, current_time_line]

def animate(i):
    """애니메이션의 각 프레임을 업데이트합니다."""
    # 표시할 데이터의 윈도우 정의
    if i < window_size:
        start = 0
        end = window_size
    else:
        start = i - window_size
        end = i

    # 현재 프레임이 끝을 초과하지 않도록 조정
    if end > len(t):
        end = len(t)
        start = end - window_size if end - window_size >= 0 else 0

    # 오실레이터 파형 업데이트
    line_osc.set_data(t[start:end], oscillator_wave[start:end])

    # 주입된 파형 업데이트
    line_inj.set_data(t[start:end], injected_wave[start:end])

    # 결합된 파형 업데이트
    line_combined.set_data(t[start:end], combined_wave[start:end])

    # 현재 시간 선 업데이트 (시퀀스로 변경)
    current_x = t[i] if i < len(t) else t[-1]
    current_time_line.set_xdata([current_x, current_x])

    return [line_osc, line_inj, line_combined, current_time_line]

# 애니메이션 생성
ani = animation.FuncAnimation(
    fig, animate, frames=frames,
    init_func=init, blit=False, interval=20, repeat=False
)

# 애니메이션을 파일로 저장 (선택 사항)
ani.save('self_injection_locking_basic.gif', writer='imagemagick')

plt.show()
