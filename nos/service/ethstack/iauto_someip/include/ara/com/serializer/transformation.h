#ifndef SERIALIZE_H_
#define SERIALIZE_H_

#include <type_traits>
#include <ara/com/serializer/transformation_reflection.h>


#define ENUM_TO_INTEGER_DEFINE(TYPE) \
    template<typename T> \
    inline std::enable_if_t<sizeof(T) == sizeof(TYPE), TYPE&> \
    _enum_to_integer(T& value) { return *(reinterpret_cast<TYPE *>(&value)); }
// TLV
#define STRUCTURE_TLV_DECLARE() \
    template<typename T> struct _TlvStructure : public std::false_type {}

#define STRUCTURE_TLV_DEFINE(STRUCT_NAME, ...) \
    template<> struct _TlvStructure<STRUCT_NAME> : public std::true_type { \
        static constexpr decltype(auto) make() { return std::make_tuple(__VA_ARGS__); } \
    }

template<typename T> struct _TlvStructure;
template<typename T> struct _is_tlv : public _TlvStructure<T> {};
template<size_t I, typename T>
constexpr decltype(auto) _tlv_get()
{
    using M = _TlvStructure<typename std::remove_cv_t<std::remove_reference_t<T>>>;
    return std::get<I>(M::make());
}


template<typename Serialize>
struct OutputStream {
    using DeployType = typename Serialize::DeployType;
    static constexpr bool DeployEnable = !std::is_void<DeployType>::value;
    static constexpr bool TlvEnable = Serialize::TlvEnable;
    ENUM_TO_INTEGER_DEFINE(uint8_t);
    ENUM_TO_INTEGER_DEFINE(uint16_t);
    ENUM_TO_INTEGER_DEFINE(uint32_t);
    ENUM_TO_INTEGER_DEFINE(uint64_t);
    OutputStream(Serialize& serialize):mSerialize(serialize){}

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_integral<T>::value, int>
    write(const T& value) {
        return mSerialize.writeValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_floating_point<T>::value, int>
    write(const T& value) {
        return mSerialize.writeValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_enum<T>::value, int>
    write(const T& value) {
        return mSerialize.writeValue(_enum_to_integer(const_cast<T&>(value)));
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_string<T>::value, int>
    write(const T& value) {
        return mSerialize.writeValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_string<T>::value, int>
    write(const T& value) {
        return mSerialize.writeValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_array<T>::value && _is_array_spcial<T>::value, int>
    write(const T& value) {
        int ret = mSerialize.writeArrayBegin();
        if (!ret) {
            ret = mSerialize.writeValue(reinterpret_cast<const uint8_t*>(value.data()), value.size());
            mSerialize.writeArrayEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable &&  _is_array<T>::value && !_is_array_spcial<T>::value, int>
    write(const T& value) {
        int ret = mSerialize.writeArrayBegin();
        if (!ret) {
            for (auto it: value) {
                ret = write(it);
                if (ret) {
                    break;
                }
            }
            mSerialize.writeArrayEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_vector<T>::value && _is_vector_spcial<T>::value, int>
    write(const T& value) {
        int ret = 0;
        ret = mSerialize.writeVectorBegin();
        if (!ret) {
            ret = mSerialize.writeValue(value);
            mSerialize.writeVectorEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable &&  _is_vector<T>::value && !_is_vector_spcial<T>::value, int>
    write(const T& value) {
        int ret = 0;
        mSerialize.writeVectorBegin();
        for (auto it: value) {
            ret = write(it);
            if (ret) {
                break;
            }
        }
        mSerialize.writeVectorEnd();
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_map<T>::value, int>
    write(const T& value) {
        int ret = 0;
        mSerialize.writeVectorBegin();
        for (auto it : value) {
            ret |= write(it.first);
            ret |= write(it.second);
            if (ret) {
                break;
            }
        }
        mSerialize.writeVectorEnd();
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_unordered_map<T>::value, int>
    write(const T& value) {
        int ret = 0;
        mSerialize.writeVectorBegin();
        for (auto it : value) {
            ret |= write(it.first);
            ret |= write(it.second);
            if (ret) {
                break;
            }
        }
        mSerialize.writeVectorEnd();
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value, int>
    write(const T& value) {
        int ret;
        ret = value.index();
        if (-1 == ret) {
            ret = 1;
        }
        else {
            ret = _write(ret, value);
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value, int>
    write(const T& value) {
        int ret = 0;
        mSerialize.writeStructBegin();
        ret = _write(value);
        mSerialize.writeStructEnd();
        return ret;
    }

    template<size_t I = 0, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value && I < _reflection_size<T>(), int>
    _write(const T& value) {
        int ret = 0;
        ret = write(_reflection_get<I>(value));
        if (!ret) {
            _write<I+1, T>(value);
            ret = 0;
        }
        return ret;
    }

    template< typename T, size_t I = 0, int R = 0>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value && I < _ara_variant_size<T>(), int>
    _write(uint32_t index, const T& value) {
        int ret = 0;
        if (index == I) {
            ret = mSerialize.writeUnionBegin(index);
            if (!ret) {
                try{
                    write(ara::core::get<I>(value));
                    mSerialize.writeUnionEnd();
                }
                catch (const ara::core::bad_variant_access& ex) {
                    ret = 1;
                    mSerialize.writeUnionEnd();
                   _write<T, _ara_variant_size<T>(), 1>(index, value);
                }
            }
        }
        else {
            ret = _write<T, I+1>(index, value);
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value, int>
    write_struct_as_args(const T& value) {
        return _write(value);
    }

    template<bool dummy = true, typename Args>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    serialize(const Args& args) {
       return write(args);
    }

    template<bool dummy = true, typename CurArgs, typename ... RestArgs>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    serialize(const CurArgs& current, const RestArgs& ...rest) {
       int ret = 0;
       ret =  write(current);
       if (!ret) {
           serialize(rest...);
           ret = 0;
       }
       return ret;
    }

    template<bool dummy = true, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    write_args(const T& args) {
        return write(args);
    }

    template<bool dummy = true, typename... ARGS>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    write_args(const std::tuple<ARGS ...>& args) {
        return _write_args(args);
    }

    template<size_t I = 0, typename ...ARGS>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && (I < sizeof ...(ARGS)), int>
    _write_args(const std::tuple<ARGS ...>& args) {
        int ret = 0;
        ret = write(std::get<I>(args));
        if (!ret) {
            _write_args<I + 1, ARGS...>(args);
            ret = 0;
        }
        return ret;
    }

    /* partial specialization */
    template<size_t I, typename ...ARGS>
    inline std::enable_if_t<I == sizeof ...(ARGS), void>
    _write_args(const std::tuple<ARGS ...>& args) {(void)args;}

    template<size_t I, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value && I == _reflection_size<T>(), void>
    _write(const T& value) {(void)value;}

    template<typename T, size_t I = 0, int R = 0>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value && I == _ara_variant_size<T>(), int>
    _write(uint32_t index, const T& value) {(void)index; (void)value; return R; }

    Serialize& mSerialize;
};


template<typename DeSerialize>
struct InputStream {
    using DeployType = typename DeSerialize::DeployType;
    static constexpr bool DeployEnable = !std::is_void<DeployType>::value;
    static constexpr bool TlvEnable = DeSerialize::TlvEnable;
    ENUM_TO_INTEGER_DEFINE(uint8_t);
    ENUM_TO_INTEGER_DEFINE(uint16_t);
    ENUM_TO_INTEGER_DEFINE(uint32_t);
    ENUM_TO_INTEGER_DEFINE(uint64_t);
    InputStream(DeSerialize& serialize):mDeSerialize(serialize){}

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_integral<T>::value, int>
    read(T& value) {
        return mDeSerialize.readValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_floating_point<T>::value, int>
    read(T& value) {
        return mDeSerialize.readValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && std::is_enum<T>::value, int>
    read(T& value) {
        return mDeSerialize.readValue( _enum_to_integer(value));
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_string<T>::value, int>
    read(T& value) {
        return mDeSerialize.readValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_string<T>::value, int>
    read(T& value) {
        return mDeSerialize.readValue(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_array<T>::value && _is_array_spcial<T>::value, int>
    read(T& value) {
        int ret = mDeSerialize.readArrayBegin();
        if (!ret) {
            ret = mDeSerialize.readValue(reinterpret_cast<uint8_t*>(value.data()), value.size());
            mDeSerialize.readArrayEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_array<T>::value && !_is_array_spcial<T>::value, int>
    read(T& value) {
        typename T::value_type array_value;
        int ret = mDeSerialize.readArrayBegin();
        if (!ret) {
            for (std::size_t index = 0; index < _array_size<T>(); index++) {
                ret = read(array_value);
                if (!ret) {
                    value[index] = array_value; // TODO: move value?
                }
                else {
                    break;
                }
            }
            mDeSerialize.readArrayEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_vector<T>::value && _is_vector_spcial<T>::value, int>
    read(T& value) {
        int ret;
        uint32_t size = 0; /* TODO */
        ret = mDeSerialize.readVectorBegin();
        if (!ret) {
            ret = mDeSerialize.readValue(value, size);
            mDeSerialize.readVectorEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_vector<T>::value && !_is_vector_spcial<T>::value, int>
    read(T& value) {
        int ret;
        typename T::value_type vector_value;
        ret = mDeSerialize.readVectorBegin();
        if (ret != 0) {
        }
        else {
            ret = read(vector_value);
            while(!ret) {
                value.push_back(std::move(vector_value));
                ret = read(vector_value);
                continue;
            }
            mDeSerialize.readVectorEnd();
            ret = (ret==3)?0:1;
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_map<T>::value, int>
    read(T& value) {
        int ret;
        typename T::key_type map_key;
        typename T::mapped_type map_value;
        using map_element_t = typename std::map<typename T::key_type, typename T::mapped_type>::value_type;
        ret = mDeSerialize.readVectorBegin();
        if (ret != 0) {
        }
        else {
            ret  = read(map_key);
            ret |= read(map_value);
            while(!ret) {
                value.insert(map_element_t(std::move(map_key), std::move(map_value)));
                ret |= read(map_key);
                ret |= read(map_value);
                continue;
            }
            mDeSerialize.readVectorEnd();
            ret = (ret==3)?0:1;
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_unordered_map<T>::value, int>
    read(T& value) {
        int ret;
        typename T::key_type map_key;
        typename T::mapped_type map_value;
        using map_element_t = typename std::unordered_map<typename T::key_type, typename T::mapped_type>::value_type;
        ret = mDeSerialize.readVectorBegin();
        if (ret != 0) {
        }
        else {
            ret |= read(map_key);
            ret |= read(map_value);
            while(!ret) {
                value.insert(map_element_t(std::move(map_key), std::move(map_value)));
                ret |= read(map_key);
                ret |= read(map_value);
                continue;
            }
            mDeSerialize.readVectorEnd();
            ret = (ret==3)?0:1;
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value, int>
    read(T& value) {
        int ret;
        ret = mDeSerialize.readStructBegin();
        if (ret) {
        }
        else {
            ret = _read(value);
            mDeSerialize.readStructEnd();
        }
        return ret;
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value, int>
    read_struct_as_args(T& value) {
        return _read(value);
    }

    template<typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value, int>
    read(T& value) {
        int ret;
        uint32_t index;
        ret = mDeSerialize.readUnionBegin(index);
        if (ret) {
        }
        else {
           ret = _read(index, value);
           mDeSerialize.readUnionEnd();
        }
        return ret;
    }

    template<size_t I = 0, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && I < _reflection_size<T>(), int>
    _read(T& value) {
        int ret;
        ret = read(_reflection_get<I>(value));
        if (ret == 0 || ret == 3) {
            _read<I+1, T>(value);
            ret = 0;
        }
        else {
            ret = 1;
        }
        return ret;
    }

    template< typename T, size_t I = 0>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value && I < _ara_variant_size<T>(), int>
    _read(uint32_t index, T& value) {
        int ret = 0;
        if (index == I) {
            using variant_type = typename _type_traits<T>::template Arg<I>::type;
            variant_type variant_value;
            ret = read(variant_value);
            if (!ret) {
                /* value.emplace<I>(variant_value); */
                value = variant_value;
            }
        }
        else {
            _read<T, I+1>(index, value);
        }
        return ret;
    }

    template<bool dummy = true, typename Args>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    deserialize(Args& args) {
       return read(args);
    }

    template<bool dummy = true, typename CurArgs, typename ... RestArgs>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    deserialize(CurArgs& current, RestArgs& ...rest) {
       int ret;
       ret = read(current);
       if (ret == 0 || ret == 3) {
           deserialize(rest...);
           ret = 0;
       }
       else {
           ret = 1;
       }
       return ret;
    }

    template<bool dummy = true, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    read_args(T& args) {
       return read(args);
    }

    template<bool dummy = true, typename... ARGS>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && dummy, int>
    read_args(const std::tuple<ARGS ...>& args) {
       return _read_args(args);
    }

    template<size_t I = 0, typename ...ARGS>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && (I < sizeof ...(ARGS)), int>
    _read_args(const std::tuple<ARGS ...>& args) {
        int ret;
        ret = read(std::get<I>(args));
        if (ret == 0 || ret == 3) {
            _read_args<I + 1, ARGS...>(args);
           ret = 0;
        }
        else {
            ret = 1;
        }
        return ret;
    }

    /* partial specialization */
    template<size_t I, typename ...ARGS>
    inline std::enable_if_t<I == sizeof ...(ARGS), void>
    _read_args(const std::tuple<ARGS ...>& args) {(void)args;}

    template<size_t I, typename T>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_refection<T>::value && I == _reflection_size<T>(), void>
    _read(T& value) {(void)value;}

    template< typename T, size_t I = 0>
    inline std::enable_if_t<!DeployEnable && !TlvEnable && _is_ara_variant<T>::value && I == _ara_variant_size<T>(), void>
    _read(uint32_t index, T& value) {(void)index; (void)value;}

    DeSerialize& mDeSerialize;
};
#endif // SERIALIZE_H_
