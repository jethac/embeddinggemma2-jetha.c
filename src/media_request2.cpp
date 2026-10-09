/* JSON/media boundary only; inference graph and serving remain C. MIT. */
extern "C" {
#include "media2.h"
}
#include "nlohmann/json.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

static std::vector<unsigned char> decode_base64(std::string_view text) {
    if (text.rfind("data:", 0) == 0) {
        size_t split = text.find(";base64,");
        if (split == std::string::npos) throw std::runtime_error("media data URL must be base64 encoded");
        text.remove_prefix(split + 8);
    }
    if (text.empty() || text.size() % 4) throw std::runtime_error("invalid base64 media");
    static const auto lookup = [] {
        std::array<signed char, 256> table;
        table.fill(-1);
        const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < 64; i++)
            table[static_cast<unsigned char>(alphabet[i])] = static_cast<signed char>(i);
        return table;
    }();
    std::vector<unsigned char> bytes;
    bytes.reserve(text.size() / 4 * 3);
    // Only the final quartet can contain padding. Decode the bulk directly
    // into its bounded output, leaving the existing padding rules below.
    size_t bulk = text.size() - 4;
    bytes.resize(bulk / 4 * 3);
    for (size_t i = 0, out = 0; i < bulk; i += 4, out += 3) {
        int a = lookup[static_cast<unsigned char>(text[i])];
        int b = lookup[static_cast<unsigned char>(text[i + 1])];
        int c = lookup[static_cast<unsigned char>(text[i + 2])];
        int d = lookup[static_cast<unsigned char>(text[i + 3])];
        if ((a | b | c | d) < 0) throw std::runtime_error("invalid base64 media");
        unsigned value = (unsigned)a << 18 | (unsigned)b << 12 | (unsigned)c << 6 | (unsigned)d;
        bytes[out] = (unsigned char)(value >> 16);
        bytes[out + 1] = (unsigned char)(value >> 8);
        bytes[out + 2] = (unsigned char)value;
    }
    for (size_t i = bulk; i < text.size(); i += 4) {
        unsigned value = 0;
        int padding = 0;
        for (size_t j = 0; j < 4; j++) {
            char c = text[i + j];
            if (c == '=') {
                if (j < 2 || i + 4 != text.size()) throw std::runtime_error("invalid base64 padding");
                ++padding;
                value <<= 6;
            } else {
                int p = lookup[static_cast<unsigned char>(c)];
                if (padding || p < 0) throw std::runtime_error("invalid base64 media");
                value = (value << 6) | (unsigned)p;
            }
        }
        if (padding > 2) throw std::runtime_error("invalid base64 padding");
        bytes.push_back((unsigned char)(value >> 16));
        if (padding < 2) bytes.push_back((unsigned char)(value >> 8));
        if (!padding) bytes.push_back((unsigned char)value);
    }
    return bytes;
}

static std::string encode_base64(const float *data, size_t count) {
    static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const auto *bytes = reinterpret_cast<const unsigned char *>(data);
    size_t len = count * sizeof(float);
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned n = (unsigned)bytes[i] << 16;
        if (i + 1 < len) n |= (unsigned)bytes[i + 1] << 8;
        if (i + 2 < len) n |= bytes[i + 2];
        out.push_back(alphabet[(n >> 18) & 63]); out.push_back(alphabet[(n >> 12) & 63]);
        out.push_back(i + 1 < len ? alphabet[(n >> 6) & 63] : '=');
        out.push_back(i + 2 < len ? alphabet[n & 63] : '=');
    }
    return out;
}

extern "C" bool ei_multimodal_request(ei_engine *e, const char *body, size_t body_len,
    bool openai, size_t max_batch, char **response, char *err, size_t err_len) {
    try {
        json request = json::parse(body, body + body_len);
        int dimensions = request.value("dimensions", 768);
        if (dimensions != 128 && dimensions != 256 && dimensions != 512 && dimensions != 768)
            throw std::runtime_error("dimensions must be 128, 256, 512, or 768");
        std::string encoding = request.value("encoding_format", std::string("float"));
        if (encoding != "float" && encoding != "base64") throw std::runtime_error("unsupported encoding_format");
        std::string model = request.value("model", std::string("embeddinggemma-2"));
        json input = request.at("input");
        if (input.is_object()) input = json::array({input});
        if (!input.is_array() || input.empty() || input.size() > max_batch)
            throw std::runtime_error("input must be an object or a bounded array of objects");
        json result = openai ? json{{"object", "list"}, {"data", json::array()}, {"model", model}}
                             : json{{"embeddings", json::array()}};
        size_t total_tokens = 0;
        double total_encoder = 0, total_backbone = 0;
        std::vector<std::vector<ei_media_part>> items(input.size());
        std::vector<std::vector<std::vector<unsigned char>>> owned(input.size());
        std::vector<const ei_media_part *> pointers(input.size());
        std::vector<size_t> counts(input.size()), token_counts(input.size());
        std::vector<float> embeddings(input.size() * 768);
        for (size_t index = 0; index < input.size(); index++) {
            const json &content = input[index].at("content");
            if (!content.is_array() || content.empty() || content.size() > 64)
                throw std::runtime_error("content must contain 1..64 parts");
            auto &parts = items[index];
            auto &buffers = owned[index];
            parts.resize(content.size());
            buffers.resize(content.size());
            for (size_t i = 0; i < content.size(); i++) {
                const json &part = content[i];
                std::string type = part.at("type").get<std::string>();
                if (type == "text") {
                    const auto &text = part.at("text").get_ref<const std::string &>();
                    parts[i] = {EI_PART_TEXT, reinterpret_cast<const unsigned char *>(text.data()), text.size(), 0};
                } else {
                    ei_part_type kind;
                    if (type == "image") kind = EI_PART_IMAGE;
                    else if (type == "audio") kind = EI_PART_AUDIO;
                    else if (type == "video") kind = EI_PART_VIDEO;
                    else throw std::runtime_error("content type must be text, image, audio, or video");
                    buffers[i] = decode_base64(part.at("data").get_ref<const std::string &>());
                    float fps = part.value("fps", 1.0f);
                    if (!std::isfinite(fps) || fps <= 0 || fps > 30) throw std::runtime_error("fps must be >0 and <=30");
                    parts[i] = {kind, buffers[i].data(), buffers[i].size(), fps};
                }
            }
            pointers[index] = parts.data();
            counts[index] = parts.size();
        }
        if (ei_engine_media_batch_enabled(e) && input.size() > 1) {
            for (size_t index = 0; index < input.size(); index += 256) {
                size_t count = std::min(size_t(256), input.size() - index);
                double encoder = 0, backbone = 0;
                if (!ei_engine_embed_parts_batch(e, pointers.data() + index, counts.data() + index,
                    count, embeddings.data() + index * 768, token_counts.data() + index,
                    &encoder, &backbone, err, err_len)) return false;
                total_encoder += encoder;
                total_backbone += backbone;
            }
        } else {
            for (size_t index = 0; index < input.size(); index++) {
                double encoder = 0, backbone = 0;
                if (!ei_engine_embed_parts(e, pointers[index], counts[index],
                    embeddings.data() + index * 768, &token_counts[index],
                    &encoder, &backbone, err, err_len)) return false;
                total_encoder += encoder;
                total_backbone += backbone;
            }
        }
        for (size_t index = 0; index < input.size(); index++) {
            float *embedding = embeddings.data() + index * 768;
            double energy = 0;
            for (int j = 0; j < dimensions; j++) energy += (double)embedding[j] * embedding[j];
            if (!(energy > 0)) throw std::runtime_error("zero embedding after truncation");
            float scale = (float)(1.0 / std::sqrt(energy));
            for (int j = 0; j < dimensions; j++) embedding[j] *= scale;
            json vector = encoding == "base64" ? json(encode_base64(embedding, (size_t)dimensions))
                : json(std::vector<float>(embedding, embedding + dimensions));
            if (openai) result["data"].push_back({{"object", "embedding"}, {"index", index}, {"embedding", vector}});
            else result["embeddings"].push_back(vector);
            total_tokens += token_counts[index];
        }
        result["usage"] = {{"prompt_tokens", total_tokens}, {"total_tokens", total_tokens}};
        fprintf(stderr, "multimodal request: %zu inputs, %zu tokens, encoder %.2f ms, backbone %.2f ms\n",
                input.size(), total_tokens, total_encoder, total_backbone);
        std::string text = result.dump();
        *response = static_cast<char *>(malloc(text.size() + 1));
        if (!*response) throw std::runtime_error("out of memory");
        memcpy(*response, text.c_str(), text.size() + 1);
        return true;
    } catch (const std::exception &ex) {
        snprintf(err, err_len, "%s", ex.what());
        return false;
    }
}
