import polars as pl
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

opcodes = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    # 0x0C - 0x0F are unused
    0x10: "LT",
    0x11: "GT",
    0x12: "SLT",
    0x13: "SGT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x16: "AND",
    0x17: "OR",
    0x18: "XOR",
    0x19: "NOT",
    0x1A: "BYTE",
    0x1B: "SHL",
    0x1C: "SHR",
    0x1D: "SAR",
    0x20: "KECCAK256",
    # 0x21 - 0x2F are unused
    0x30: "ADDRESS",
    0x31: "BALANCE",
    0x32: "ORIGIN",
    0x33: "CALLER",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY",
    0x38: "CODESIZE",
    0x39: "CODECOPY",
    0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE",
    0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH",
    0x41: "COINBASE",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x44: "DIFFICULTY",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x48: "BASEFEE",
    0x50: "POP",
    0x51: "MLOAD",
    0x52: "MSTORE",
    0x53: "MSTORE8",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x58: "GETPC",
    0x59: "MSIZE",
    0x5A: "GAS",
    0x5B: "JUMPDEST",
    # 0x5C - 0x5F are unused
    **{i: f"PUSH{i - 0x5F}" for i in range(0x60, 0x80)},
    **{i: f"DUP{i - 0x7F}" for i in range(0x80, 0x90)},
    **{i: f"SWAP{i - 0x8F}" for i in range(0x90, 0xA0)},
    **{i: f"LOG{i - 0xA0}" for i in range(0xA0, 0xA5)},
    # 0xA5 - 0xAF are unused
    # Opcodes from 0xB0 to 0xBA are tentative for EIP 615
    # 0xBB - 0xE0 are unused
    # 0xE1 - 0xEF are referenced only in pyethereum
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    # 0xF6 - 0xF9 are unused
    0xFA: "STATICCALL",
    # 0xFB is unused
    # 0xFC is not in the yellow paper
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}

def disasm(code):
    result = []
    pc = 0
    while pc < len(code):
        op = code[pc]
        opcode = opcodes.get(op, f"unknown_{hex(op)}")
        result.append(opcode)
        if 0x60 <= op <= 0x7f:  # PUSH1 to PUSH32
            push_size = op - 0x5f
            push_value = code[pc + 1 : pc + 1 + push_size]
            result.append(hex(int.from_bytes(push_value, "big")))
            pc += push_size
        pc += 1
    return " ".join(result)

def process_chunk(chunk):
    return chunk.with_columns([
        pl.col("code").map_elements(disasm).alias("disassembled_code")
    ])

def format_eth_address(address):
    return '0x' + address.hex()

def parallel_disassemble(df, num_workers=11):
    chunk_size = len(df) // num_workers
    df_chunks = [df.slice(i * chunk_size, chunk_size) for i in range(num_workers)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in df_chunks]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
    return pl.concat(results)

if __name__ == '__main__':
    data_path = './data/*.parquet'
    parquet_files = glob.glob(data_path)
    os.makedirs('./results', exist_ok=True)

    total_original_rows = 0
    total_filtered_rows = 0
    all_results = []

    for parquet_file in parquet_files:
        all_contracts_df = pl.read_parquet(parquet_file)
        original_row_count = len(all_contracts_df)
        total_original_rows += original_row_count

        disassembled_df = parallel_disassemble(all_contracts_df)
        contracts_using_pairing = disassembled_df.filter(
            pl.col("disassembled_code").str.contains(r'PUSH1 0x8 GAS STATICCALL')
        ).with_columns([
            pl.col("contract_address").map_elements(format_eth_address, return_dtype=pl.Utf8).alias("formatted_address")
        ])

        filtered_row_count = len(contracts_using_pairing)
        total_filtered_rows += filtered_row_count
        all_results.append(contracts_using_pairing)

        print(f"File {parquet_file}: Original rows = {original_row_count}, Filtered rows = {filtered_row_count}")

    combined_results = pl.concat(all_results)

    clean_combined_results = combined_results.select(['disassembled_code', 'formatted_address'])

    # Save the complete and clean versions
    combined_results.write_parquet('./results/output.parquet')
    clean_combined_results.write_parquet('./results/clean_output.parquet')

    print(f"Total original rows = {total_original_rows}, Total filtered rows = {total_filtered_rows}")