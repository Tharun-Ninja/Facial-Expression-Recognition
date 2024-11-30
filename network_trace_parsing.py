import re

ip_to_node = {
    "10.1.1.1": "A",
    "10.1.2.1": "B",
    "10.1.3.1": "C",
    "10.1.4.1": "D",
    "10.1.5.1": "E",
    "10.1.6.1": "F",
    "10.1.7.1": "G",
    "10.1.8.1": "R1",
    "10.1.9.1": "R2",
    "10.1.10.1": "R3",
    "10.1.11.1": "R4"
}

routing_table = {
    "Source": ["A", "B", "C", "D", "E", "F", "G", "R1", "R2", "R3", "R4"],
    "Destination": ["A", "B", "C", "D", "E", "F", "G", "R1", "R2", "R3", "R4"],
    "R1": ["-", "R1", "R1", "R1", "R1", "R1", "R1", "-", "-", "-", "-"],
    "R2": ["-", "-", "R1", "R1", "R1", "R4", "R2", "R2", "-", "R2", "R2"],
    "R3": ["R3", "R1", "-", "R3", "R3", "R3", "R3", "C", "R1", "-", "R3"],
    "R4": ["R4", "R2", "R3", "R3", "R4", "R2", "G", "R1", "R1", "R1", "-"]
}

with open('network_tracing.tr', 'r') as f:
    for line in f:
        if 'Enqueue' in line or 'Dequeue' in line:
            src_ip, dst_ip = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', line)
            src_node = ip_to_node[src_ip]
            dst_node = ip_to_node[dst_ip]
            if src_node in routing_table["Source"] and dst_node in routing_table["Destination"]:
                next_hop = routing_table[src_node][routing_table["Destination"].index(dst_node)]
                print(f'Source: {src_node}, Destination: {dst_node}, Next Hop: {next_hop}')
            else:
                print(f'Source: {src_node}, Destination: {dst_node}, Next Hop: Unknown')