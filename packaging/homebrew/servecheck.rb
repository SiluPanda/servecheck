class Servecheck < Formula
  desc "15-second conformance probe and offline GGUF/Jinja2 doctor for local LLM servers"
  homepage "https://github.com/SiluPanda/servecheck"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/SiluPanda/servecheck/releases/download/v0.1.0/servecheck_v0.1.0_darwin_arm64.tar.gz"
      sha256 "b9236cb88e887deebd130255b6e9fea9d68afb357daf7b16385fffd948cbcb72"
    else
      url "https://github.com/SiluPanda/servecheck/releases/download/v0.1.0/servecheck_v0.1.0_darwin_amd64.tar.gz"
      sha256 "7724c73d319249b5d302023babc728528e6ac52ab5c10badd6d2ee78d1e9ca9f"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/SiluPanda/servecheck/releases/download/v0.1.0/servecheck_v0.1.0_linux_arm64.tar.gz"
      sha256 "82a345eab6d7e969d4295d1fcff940268864da8880574805dc0744ee1f2b9afe"
    else
      url "https://github.com/SiluPanda/servecheck/releases/download/v0.1.0/servecheck_v0.1.0_linux_amd64.tar.gz"
      sha256 "6988db4c8ea3bfab49231641fc19976194e2b4af9973ce20d5bc73d45445c0aa"
    end
  end

  def install
    bin.install "servecheck"
  end

  test do
    assert_match "servecheck v#{version}", shell_output("#{bin}/servecheck version")
  end
end
