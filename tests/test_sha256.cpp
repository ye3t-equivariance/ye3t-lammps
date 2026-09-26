#include "ye3t_sha256.h"

#include <iostream>
#include <stdexcept>
#include <string>

using YE3T_LAMMPS::sha256_string;
using YE3T_LAMMPS::SHA256Builder;

namespace {

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

}    // namespace

int main()
{
  try {
    require(sha256_string("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "empty SHA-256 vector mismatch");
    require(sha256_string("abc") ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "abc SHA-256 vector mismatch");

    SHA256Builder incremental;
    incremental.update("a");
    incremental.update("b");
    incremental.update("c");
    require(incremental.finish() == sha256_string("abc"), "incremental SHA-256 mismatch");

    bool finish_rejected = false;
    try {
      incremental.finish();
    } catch (const std::logic_error &) {
      finish_rejected = true;
    }
    require(finish_rejected, "a second SHA-256 finish was accepted");

    bool update_rejected = false;
    try {
      incremental.update("d");
    } catch (const std::logic_error &) {
      update_rejected = true;
    }
    require(update_rejected, "an update after SHA-256 finish was accepted");

    SHA256Builder null_input;
    bool null_rejected = false;
    try {
      null_input.update(nullptr, 1);
    } catch (const std::invalid_argument &) {
      null_rejected = true;
    }
    require(null_rejected, "nonempty null SHA-256 input was accepted");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "SHA-256 known-vector and streaming tests passed.\n";
  return 0;
}
