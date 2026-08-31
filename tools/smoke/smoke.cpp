// Host smoke tool. Not a deliverable: exists so CI can catch a broken engine
// patch in seconds, on the same ruy/NEON code path devices use.
//
// Usage: smoke <model-config.yml> [<second-config.yml>] < text-lines
//   One input line = one source text, translated as a batch (mirrors the JNI
//   surface). With a second config, translates via pivot (first -> second).
#include <iostream>
#include <string>
#include <vector>

#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

int main(int argc, char *argv[]) {
  using namespace marian::bergamot;
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: smoke <model-config.yml> [<second-config.yml>] < text-lines\n";
    return 2;
  }

  BlockingService service{BlockingService::Config{}};
  auto makeModel = [&](const char *path) {
    return std::make_shared<TranslationModel>(parseOptionsFromFilePath(path));
  };
  auto model = makeModel(argv[1]);

  std::vector<std::string> sources;
  for (std::string line; std::getline(std::cin, line);) sources.push_back(line);
  std::vector<ResponseOptions> options(sources.size());

  std::vector<Response> responses =
      (argc == 3) ? service.pivotMultiple(model, makeModel(argv[2]), std::move(sources), options)
                  : service.translateMultiple(model, std::move(sources), options);

  for (auto &response : responses) std::cout << response.target.text << "\n";
  return 0;
}
