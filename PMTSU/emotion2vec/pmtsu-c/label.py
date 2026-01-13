#它把 IEMOCAP 官方原始评估文件（EmoEvaluation/*.txt）里“谁说了哪句话、什么情感、VAD 连续值”抽出来，按对应 wav 文件的自然顺序写成一行一个标签的简洁 TSV：
import os
import codecs
import argparse
from collections import defaultdict
import glob

def generate_iemocap_labels(data_path, output_file):
    """
    生成简洁格式的IEMOCAP标签文件
    输出格式: Sentence_ID Emotion Valence Arousal Dominance
    """
    # 数据结构
    label_dict = {}          # 存储所有标签 {sentence_id: (emotion, v, a, d)}
    audio_files = []          # 存储音频文件路径（保持顺序）
    missing_labels = []       # 存储缺失标签的ID
    emotion_stats = defaultdict(int)  # 情感类别统计
    
    # 配置参数
    valid_emotions = {'neu', 'ang', 'hap', 'sad'}  # 只处理这四种情感
    total_sessions = 5        # IEMOCAP共5个session

    # 打印初始化信息
    print("=" * 80)
    print("IEMOCAP Label Generator - Simplified Format")
    print("=" * 80)
    print(f"🔍 Dataset Path: {data_path}")
    print(f"📝 Output File: {output_file}")
    print("-" * 80)

    # 阶段1: 收集所有音频文件（保持原始顺序）
    print("\n🔊 Collecting audio files in natural order...")
    for session_id in range(1, total_sessions + 1):
        session_name = f"Session{session_id}"
        wav_dir = os.path.join(data_path, session_name, 'sentences', 'wav')
        
        if not os.path.exists(wav_dir):
            print(f"⚠️ Warning: Missing directory - {wav_dir}")
            continue
            
        # 递归查找.wav文件并排序
        session_files = glob.glob(os.path.join(wav_dir, '**', '*.wav'), recursive=True)
        session_files.sort()  # 保持自然顺序
        audio_files.extend(session_files)
        print(f"  {session_name}: found {len(session_files)} .wav files")

    total_audio_files = len(audio_files)
    print(f"✅ Collected {total_audio_files} audio files in total")
    print("-" * 80)

    # 阶段2: 解析情感标签
    print("\n🏷️ Parsing emotion labels from evaluation files...")
    for session_id in range(1, total_sessions + 1):
        session_name = f"Session{session_id}"
        eval_dir = os.path.join(data_path, session_name, 'dialog', 'EmoEvaluation')
        
        if not os.path.exists(eval_dir):
            print(f"⚠️ Warning: Missing evaluation dir - {eval_dir}")
            continue
            
        # 处理每个标注文件
        for eval_file in os.listdir(eval_dir):
            if not eval_file.endswith('.txt'):
                continue
                
            file_path = os.path.join(eval_dir, eval_file)
            with codecs.open(file_path, 'r', encoding='utf-8') as f:
                process_block = False
                
                for line in f:
                    line = line.strip()
                    
                    if not line:
                        process_block = True
                        continue
                        
                    if process_block and line.startswith('['):
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                            
                        # 解析关键字段
                        sentence_id = parts[3]
                        emotion = parts[4]
                        
                        # 情感标签处理
                        if emotion == 'exc':
                            emotion = 'hap'  # 合并excited到happy
                            
                        if emotion in valid_emotions:
                            try:
                                # 直接解析VAD值（不再离散化）
                                vad_str = ''.join(parts[5:8]).replace('[', '').replace(']', '')
                                v, a, d = map(float, vad_str.split(','))
                                
                                label_dict[sentence_id] = (emotion, v, a, d)
                                emotion_stats[emotion] += 1
                            except (ValueError, IndexError) as e:
                                print(f"⚠️ Parse error in {eval_file}: {line} | Error: {str(e)}")

    print(f"✅ Parsed {len(label_dict)} valid emotion labels")
    print("-" * 80)

    # 阶段3: 按音频文件顺序生成标签
    print("\n✍️ Generating label file in audio file order...")
    with open(output_file, 'w') as f_out:
        # 写入文件头
        # f_out.write("Sentence_ID    Emotion Valence Arousal Dominance\n")
        
        # 按音频文件顺序写入标签
        matched_labels = 0
        for audio_path in audio_files:
            sentence_id = os.path.splitext(os.path.basename(audio_path))[0]
            
            if sentence_id in label_dict:
                emotion, v, a, d = label_dict[sentence_id]
                f_out.write(f"{sentence_id}\t{emotion}\t{v:.4f}\t{a:.4f}\t{d:.4f}\n")
                matched_labels += 1
            else:
                missing_labels.append(sentence_id)

    # 阶段4: 生成统计报告
    print("\n" + "=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"📊 Total Audio Files: {total_audio_files}")
    print(f"🏷️  Matched Labels: {matched_labels} ({matched_labels/total_audio_files:.1%})")
    print(f"⚠️  Missing Labels: {len(missing_labels)}")
    print("-" * 80)
    
    # 情感分布统计
    print("Emotion Distribution:")
    for emotion in sorted(emotion_stats):
        count = emotion_stats[emotion]
        print(f"  {emotion.upper()}: {count} ({count/matched_labels:.1%})")
    
    # 缺失标签示例
    if missing_labels:
        print(f"\nMissing Label Examples (first 5):")
        for label in missing_labels[:5]:
            print(f"  {label}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels)-5} more")
    
    print("-" * 80)
    print(f"✅ Successfully generated label file")
    print(f"📋 Output: {output_file}")
    print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate IEMOCAP emotion labels in simplified format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='/mnt/cxh10/database/lizr/emotion/IEMOCAP',
                       help='IEMOCAP dataset root directory')
    parser.add_argument('--output', default='/mnt/cxh10/database/lizr/emotion/emotion2vec/iemocap_downstream_main/vad.lab', help='Output label file path')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 运行标签生成器
    generate_iemocap_labels(args.data_path, args.output)