/* JSON/media boundary only; inference graph and serving remain C. MIT. */
extern "C" {
#include "media2.h"
}
#include "nlohmann/json.hpp"
#include "simdjson.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

static json materialize_json(simdjson::dom::element element) {
    using simdjson::dom::element_type;
    switch (element.type()) {
        case element_type::ARRAY: {
            json out = json::array();
            for (auto child : element.get_array()) out.push_back(materialize_json(child));
            return out;
        }
        case element_type::OBJECT: {
            json out = json::object();
            // Preserve the existing parser's last-value-wins duplicate keys.
            for (auto field : element.get_object())
                out[std::string(field.key)] = materialize_json(field.value);
            return out;
        }
        case element_type::STRING: return std::string(element.get_string().value());
        case element_type::INT64: return element.get_int64().value();
        case element_type::UINT64: return element.get_uint64().value();
        case element_type::DOUBLE: return element.get_double().value();
        case element_type::BIGINT: {
            auto number = element.get_bigint().value();
            return json::parse(number.begin(), number.end());
        }
        case element_type::BOOL: return element.get_bool().value();
        case element_type::NULL_VALUE: return nullptr;
    }
    throw std::runtime_error("unknown JSON type");
}

static json parse_media_json(const char *body, size_t len) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    // parse() copies into padded storage; HTTP buffers need no SIMD overread.
    if (!parser.parse(body, len).get(document)) return materialize_json(document);
    // Retain acceptance and error behavior for parser limits and unusual numbers.
    return json::parse(body, body + len);
}

static std::string_view base64_payload(std::string_view text) {
    if (text.rfind("data:", 0) == 0) {
        size_t split = text.find(";base64,");
        if (split == std::string::npos) throw std::runtime_error("media data URL must be base64 encoded");
        text.remove_prefix(split + 8);
    }
    return text;
}

static std::vector<unsigned char> decode_base64(std::string_view text, bool validate_only = false) {
    text = base64_payload(text);
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
    if (!validate_only) bytes.reserve(text.size() / 4 * 3);
    // Only the final quartet can contain padding. Decode the bulk directly
    // into its bounded output, leaving the existing padding rules below.
    size_t bulk = text.size() - 4;
    if (!validate_only) bytes.resize(bulk / 4 * 3);
    for (size_t i = 0, out = 0; i < bulk; i += 4, out += 3) {
        int a = lookup[static_cast<unsigned char>(text[i])];
        int b = lookup[static_cast<unsigned char>(text[i + 1])];
        int c = lookup[static_cast<unsigned char>(text[i + 2])];
        int d = lookup[static_cast<unsigned char>(text[i + 3])];
        if ((a | b | c | d) < 0) throw std::runtime_error("invalid base64 media");
        unsigned value = (unsigned)a << 18 | (unsigned)b << 12 | (unsigned)c << 6 | (unsigned)d;
        if (!validate_only) {
            bytes[out] = (unsigned char)(value >> 16);
            bytes[out + 1] = (unsigned char)(value >> 8);
            bytes[out + 2] = (unsigned char)value;
        }
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
        if (!validate_only) {
            bytes.push_back((unsigned char)(value >> 16));
            if (padding < 2) bytes.push_back((unsigned char)(value >> 8));
            if (!padding) bytes.push_back((unsigned char)value);
        }
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

static bool execute_media_json(ei_engine *e, const json &request,
    bool openai, size_t max_batch, char **response, char *err, size_t err_len) {
    try {
        int dimensions = request.value("dimensions", 768);
        if (dimensions != 128 && dimensions != 256 && dimensions != 512 && dimensions != 768)
            throw std::runtime_error("dimensions must be 128, 256, 512, or 768");
        std::string encoding = request.value("encoding_format", std::string("float"));
        if (encoding != "float" && encoding != "base64") throw std::runtime_error("unsupported encoding_format");
        std::string model = request.value("model", std::string("embeddinggemma-2"));
        const json &input = request.at("input");
        size_t input_count = input.is_object() ? 1 : input.size();
        if ((!input.is_object() && !input.is_array()) || !input_count || input_count > max_batch)
            throw std::runtime_error("input must be an object or a bounded array of objects");
        json result = openai ? json{{"object", "list"}, {"data", json::array()}, {"model", model}}
                             : json{{"embeddings", json::array()}};
        size_t total_tokens = 0;
        double total_encoder = 0, total_backbone = 0;
        std::vector<std::vector<ei_media_part>> items(input_count);
        std::vector<std::vector<std::vector<unsigned char>>> owned(input_count);
        std::vector<const ei_media_part *> pointers(input_count);
        std::vector<size_t> counts(input_count), token_counts(input_count);
        std::vector<float> embeddings(input_count * 768);
        for (size_t index = 0; index < input_count; index++) {
            const json &content = (input.is_object() ? input : input[index]).at("content");
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
        if (ei_engine_media_batch_enabled(e) && input_count > 1) {
            for (size_t index = 0; index < input_count; index += 256) {
                size_t count = std::min(size_t(256), input_count - index);
                double encoder = 0, backbone = 0;
                if (!ei_engine_embed_parts_batch(e, pointers.data() + index, counts.data() + index,
                    count, embeddings.data() + index * 768, token_counts.data() + index,
                    &encoder, &backbone, err, err_len)) return false;
                total_encoder += encoder;
                total_backbone += backbone;
            }
        } else {
            for (size_t index = 0; index < input_count; index++) {
                double encoder = 0, backbone = 0;
                if (!ei_engine_embed_parts(e, pointers[index], counts[index],
                    embeddings.data() + index * 768, &token_counts[index],
                    &encoder, &backbone, err, err_len)) return false;
                total_encoder += encoder;
                total_backbone += backbone;
            }
        }
        for (size_t index = 0; index < input_count; index++) {
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
                input_count, total_tokens, total_encoder, total_backbone);
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

extern "C" bool ei_multimodal_request(ei_engine *e, const char *body, size_t body_len,
    bool openai, size_t max_batch, char **response, char *err, size_t err_len) {
    try {
        json request = parse_media_json(body, body_len);
        return execute_media_json(e, request, openai, max_batch, response, err, err_len);
    } catch (const std::exception &ex) {
        snprintf(err, err_len, "%s", ex.what());
        return false;
    }
}

struct ei_media_request {
    json document;
    bool openai;
    size_t max_batch, key_size;
};

// A length-delimited key retains exact text/media bytes and input ordering.
// The existing cache hash only selects a bucket; equality still uses memcmp.
class key_writer {
    char *data;
    size_t capacity, count = 0;
public:
    key_writer(char *data = nullptr, size_t capacity = SIZE_MAX) : data(data), capacity(capacity) {}
    void bytes(const char *p, size_t n) {
        if (n > capacity - count) throw std::runtime_error("media cache key exceeds its bound");
        if (data && n) memcpy(data + count, p, n);
        count += n;
    }
    void number(uint64_t value) {
        char encoded[8];
        for (unsigned i = 0; i < 8; i++) encoded[i] = char(value >> (8 * i));
        bytes(encoded, sizeof encoded);
    }
    void field(std::string_view value) { number(value.size()); bytes(value.data(), value.size()); }
    size_t size() const { return count; }
};

static void canonical_key(const ei_media_request &prepared, key_writer &key, bool validate) {
    const json &request = prepared.document;
    int dimensions = request.value("dimensions", 768);
    if (dimensions != 128 && dimensions != 256 && dimensions != 512 && dimensions != 768)
        throw std::runtime_error("dimensions must be 128, 256, 512, or 768");
    std::string encoding = request.value("encoding_format", std::string("float"));
    if (encoding != "float" && encoding != "base64") throw std::runtime_error("unsupported encoding_format");
    std::string model = request.value("model", std::string("embeddinggemma-2"));
    const json &input = request.at("input");
    size_t count = input.is_object() ? 1 : input.size();
    if ((!input.is_object() && !input.is_array()) || !count || count > prepared.max_batch)
        throw std::runtime_error("input must be an object or a bounded array of objects");
    static constexpr char domain[] = "\4EG2MEDIA1";
    key.bytes(domain, sizeof domain - 1);
    key.number(prepared.openai ? 3 : 2); key.number(dimensions);
    key.field(model); key.field(encoding);
    key.number(input.is_object() ? 0 : 1); key.number(count);
    for (size_t index = 0; index < count; index++) {
        const json &content = (input.is_object() ? input : input[index]).at("content");
        if (!content.is_array() || content.empty() || content.size() > 64)
            throw std::runtime_error("content must contain 1..64 parts");
        key.number(content.size());
        for (const json &part : content) {
            std::string type = part.at("type").get<std::string>();
            if (type == "text") {
                key.number(EI_PART_TEXT); key.field(part.at("text").get_ref<const std::string &>());
            } else {
                ei_part_type kind;
                if (type == "image") kind = EI_PART_IMAGE;
                else if (type == "audio") kind = EI_PART_AUDIO;
                else if (type == "video") kind = EI_PART_VIDEO;
                else throw std::runtime_error("content type must be text, image, audio, or video");
                const auto &data = part.at("data").get_ref<const std::string &>();
                // Reuse every decoder rejection rule without allocating decoded bytes.
                if (validate) decode_base64(data, true);
                float fps = part.value("fps", 1.0f);
                if (!std::isfinite(fps) || fps <= 0 || fps > 30) throw std::runtime_error("fps must be >0 and <=30");
                uint32_t bits; memcpy(&bits, &fps, sizeof bits);
                key.number(kind); key.number(bits); key.field(base64_payload(data));
            }
        }
    }
}

extern "C" bool ei_media_request_prepare(const char *body, size_t len, bool openai, size_t max_batch,
    ei_media_request **out, char *err, size_t err_len) {
    *out = nullptr;
    try {
        auto request = std::make_unique<ei_media_request>();
        request->document = parse_media_json(body, len);
        request->openai = openai; request->max_batch = max_batch;
        key_writer key;
        canonical_key(*request, key, true);
        request->key_size = key.size();
        *out = request.release();
        return true;
    } catch (const std::exception &ex) {
        snprintf(err, err_len, "%s", ex.what());
        return false;
    }
}

extern "C" size_t ei_media_request_key_size(const ei_media_request *request) { return request->key_size; }

extern "C" bool ei_media_request_write_key(const ei_media_request *request, char *data, size_t len,
    char *err, size_t err_len) {
    try {
        key_writer key(data, len); canonical_key(*request, key, false);
        if (key.size() != len) throw std::runtime_error("media cache key size changed");
        return true;
    } catch (const std::exception &ex) {
        snprintf(err, err_len, "%s", ex.what());
        return false;
    }
}

extern "C" bool ei_media_request_execute(ei_media_request *request, ei_engine *e, char **response,
    char *err, size_t err_len) {
    return execute_media_json(e, request->document, request->openai, request->max_batch, response, err, err_len);
}
extern "C" void ei_media_request_free(ei_media_request *request) { delete request; }
