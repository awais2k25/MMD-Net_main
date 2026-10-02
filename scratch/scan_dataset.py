import os

def scan_deepfake_timit(root_path):
    stats = {
        'higher_quality': {'videos': 0, 'audio': 0, 'identities': set()},
        'lower_quality': {'videos': 0, 'audio': 0, 'identities': set()},
        'root': {'videos': 0, 'audio': 0}
    }
    
    file_list = []
    
    for root, dirs, files in os.walk(root_path):
        rel_path = os.path.relpath(root, root_path)
        
        quality = None
        if rel_path.startswith('higher_quality'):
            quality = 'higher_quality'
        elif rel_path.startswith('lower_quality'):
            quality = 'lower_quality'
        
        if quality:
            parts = rel_path.split(os.sep)
            if len(parts) > 1:
                stats[quality]['identities'].add(parts[1])
        
        for file in files:
            if file.endswith('.avi'):
                if quality:
                    stats[quality]['videos'] += 1
                else:
                    stats['root']['videos'] += 1
                file_list.append(os.path.join(root, file))
            elif file.endswith('.wav'):
                if quality:
                    stats[quality]['audio'] += 1
                else:
                    stats['root']['audio'] += 1
                file_list.append(os.path.join(root, file))
            elif file.endswith('.mp4') or file.endswith('.mov'):
                stats['root']['videos'] += 1
                file_list.append(os.path.join(root, file))

    print(f"Dataset Statistics for {root_path}:")
    for key, val in stats.items():
        if key == 'root':
            print(f"  Root: {val['videos']} videos, {val['audio']} audio files")
        else:
            print(f"  {key}: {val['videos']} videos, {val['audio']} audio files, {len(val['identities'])} identities")
    
    print("\nSample Files:")
    for f in file_list[:20]:
        print(f"  {os.path.relpath(f, root_path)}")

if __name__ == "__main__":
    scan_deepfake_timit('/home/i212809/externaldrive/awais/DeepfakeTIMIT/')
